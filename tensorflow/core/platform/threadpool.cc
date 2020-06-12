/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/platform/threadpool.h"

#define EIGEN_USE_THREADS

#include "absl/types/optional.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/platform/setround.h"
#include "tensorflow/core/platform/tracing.h"
#include <execinfo.h>

namespace tensorflow {
namespace thread {

struct EigenEnvironment {
  typedef Thread EnvThread;
  struct TaskImpl {
    std::function<void()> f;
    Context context;
    uint64 trace_id;
  };
  struct Task {
    std::unique_ptr<TaskImpl> f;
  };

  Env* const env_;
  const ThreadOptions thread_options_;
  const string name_;

  EigenEnvironment(Env* env, const ThreadOptions& thread_options,
                   const string& name)
      : env_(env), thread_options_(thread_options), name_(name) {}

  EnvThread* CreateThread(std::function<void()> f) {
    return env_->StartThread(thread_options_, name_, [=]() {
      // Set the processor flag to flush denormals to zero.
      port::ScopedFlushDenormal flush;
      // Set the processor rounding mode to ROUND TO NEAREST.
      port::ScopedSetRound round(FE_TONEAREST);
      // port::NUMASetThreadNodeAffinity(1);
      LOG(INFO) << thread_options_.numa_node;
      if (thread_options_.numa_node != port::kNUMANoAffinity) {
        port::NUMASetThreadNodeAffinity(thread_options_.numa_node);
      }
      f();
    });
  }

  Task CreateTask(std::function<void()> f) {
    uint64 id = 0;
    if (tracing::EventCollector::IsEnabled()) {
      id = tracing::GetUniqueArg();
      tracing::RecordEvent(tracing::EventCategory::kScheduleClosure, id);
    }
    return Task{
        std::unique_ptr<TaskImpl>(new TaskImpl{
            std::move(f),
            Context(ContextKind::kThread),
            id,
        }),
    };
  }

  void ExecuteTask(const Task& t) {
    WithContext wc(t.f->context);
    tracing::ScopedRegion region(tracing::EventCategory::kRunClosure,
                                 t.f->trace_id);
    t.f->f();
  }
};

ThreadPool::ThreadPool(Env* env, const string& name, int num_threads)
    : ThreadPool(env, ThreadOptions(), name, num_threads, true, nullptr) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads)
    : ThreadPool(env, thread_options, name, num_threads, true, nullptr) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads,
                       bool low_latency_hint, Eigen::Allocator* allocator) {
  CHECK_GE(num_threads, 1);
  eigen_threadpool_.reset(new Eigen::ThreadPoolTempl<EigenEnvironment>(
      num_threads, low_latency_hint,
      EigenEnvironment(env, thread_options, "tf_" + name)));
  underlying_threadpool_ = eigen_threadpool_.get();
  threadpool_device_.reset(new Eigen::ThreadPoolDevice(underlying_threadpool_,
                                                       num_threads, allocator));
}

ThreadPool::ThreadPool(thread::ThreadPoolInterface* user_threadpool) {
  underlying_threadpool_ = user_threadpool;
  threadpool_device_.reset(new Eigen::ThreadPoolDevice(
      underlying_threadpool_, underlying_threadpool_->NumThreads(), nullptr));
}

ThreadPool::~ThreadPool() {}

void ThreadPool::Schedule(std::function<void()> fn) {
  CHECK(fn != nullptr);
  // LOG(INFO) << std::to_string(port::NUMAGetThreadNodeAffinity()); // 0
  underlying_threadpool_->Schedule(std::move(fn));
}

int ThreadPool::NumShardsUsedByFixedBlockSizeScheduling(
    const int64 total, const int64 block_size) {
  if (block_size <= 0 || total <= 1 || total <= block_size ||
      NumThreads() == 1) {
    return 1;
  }
  return (total + block_size - 1) / block_size;
}

int ThreadPool::NumShardsUsedByTransformRangeConcurrently(
    const int64 block_size, const int64 total) {
  return NumShardsUsedByFixedBlockSizeScheduling(total, block_size);
}

void ThreadPool::ParallelFor(int64 total,
                             const SchedulingParams& scheduling_params,
                             const std::function<void(int64, int64)>& fn) {
  switch (scheduling_params.strategy()) {
    case SchedulingStrategy::kAdaptive: {
      if (scheduling_params.cost_per_unit().has_value()) {
        ParallelFor(total, *scheduling_params.cost_per_unit(), fn);
      }
      break;
    }
    case SchedulingStrategy::kFixedBlockSize: {
      if (scheduling_params.block_size().has_value()) {
        ParallelForFixedBlockSizeScheduling(
            total, *scheduling_params.block_size(), fn);
      }
      break;
    }
  }
}

void ThreadPool::TransformRangeConcurrently(
    const int64 block_size, const int64 total,
    const std::function<void(int64, int64)>& fn) {
  ParallelFor(total,
              SchedulingParams(SchedulingStrategy::kFixedBlockSize,
                               absl::nullopt /* cost_per_unit */, block_size),
              fn);
}


  // Calculates block size based on (1) the iteration cost and (2) parallel
  // efficiency. We want blocks to be not too small to mitigate parallelization
  // overheads; not too large to mitigate tail effect and potential load
  // imbalance and we also want number of blocks to be evenly dividable across
  // threads.
  ThreadPool::ParallelForBlock ThreadPool::CalculateParallelForBlock(
      const Eigen::Index n, const Eigen::TensorOpCost& cost,
      std::function<Eigen::Index(Eigen::Index)> block_align) const {

    typedef Eigen::TensorCostModel<Eigen::ThreadPoolDevice> CostModel;
    const double block_size_f = 1.0 / CostModel::taskSize(1, cost);
    const Eigen::Index max_oversharding_factor = 4;
    Eigen::Index block_size = Eigen::numext::mini(
        n, Eigen::numext::maxi<Eigen::Index>(
               Eigen::divup<Eigen::Index>(n, max_oversharding_factor * underlying_threadpool_->NumThreads()),
               block_size_f));
    const Eigen::Index max_block_size = Eigen::numext::mini(n, 2 * block_size);

    if (block_align) {
      Eigen::Index new_block_size = block_align(block_size);
      eigen_assert(new_block_size >= block_size);
      block_size = Eigen::numext::mini(n, new_block_size);
    }

    Eigen::Index block_count = Eigen::divup(n, block_size);

    // Calculate parallel efficiency as fraction of total CPU time used for
    // computations:
    double max_efficiency =
        static_cast<double>(block_count) /
        (Eigen::divup<int>(block_count, NumThreads()) * NumThreads());

    // Now try to increase block size up to max_block_size as long as it
    // doesn't decrease parallel efficiency.
    for (Eigen::Index prev_block_count = block_count;
         max_efficiency < 1.0 && prev_block_count > 1;) {
      // This is the next block size that divides size into a smaller number
      // of blocks than the current block_size.
      Eigen::Index coarser_block_size = Eigen::divup(n, prev_block_count - 1);
      if (block_align) {
        Eigen::Index new_block_size = block_align(coarser_block_size);
        eigen_assert(new_block_size >= coarser_block_size);
        coarser_block_size = Eigen::numext::mini(n, new_block_size);
      }
      if (coarser_block_size > max_block_size) {
        break;  // Reached max block size. Stop.
      }
      // Recalculate parallel efficiency.
      const Eigen::Index coarser_block_count = Eigen::divup(n, coarser_block_size);
      eigen_assert(coarser_block_count < prev_block_count);
      prev_block_count = coarser_block_count;
      const double coarser_efficiency =
          static_cast<double>(coarser_block_count) /
          (Eigen::divup<int>(coarser_block_count, NumThreads()) * NumThreads());
      if (coarser_efficiency + 0.01 >= max_efficiency) {
        // Taking it.
        block_size = coarser_block_size;
        block_count = coarser_block_count;
        if (max_efficiency < coarser_efficiency) {
          max_efficiency = coarser_efficiency;
        }
      }
    }

    return {block_size, block_count};
  }

void ThreadPool::ParallelForNonFixedBlockSizeScheduling(
    Eigen::Index n, const Eigen::TensorOpCost& cost,
    std::function<Eigen::Index(Eigen::Index)> block_align,
    std::function<void(Eigen::Index, Eigen::Index)> f, const void *mem_hint) {

      // Compute small problems directly in the caller thread.
    if (n <= 1 || NumThreads() == 1) {
      f(0, n);
      return;
    }

    // Compute block size and total count of blocks.
    ParallelForBlock block = CalculateParallelForBlock(n, cost, block_align);

    // Recursively divide size into halves until we reach block_size.
    // Division code rounds mid to block_size, so we are guaranteed to get
    // block_count leaves that do actual computations.
    BlockingCounter barrier(static_cast<unsigned int>(block.count));
    std::function<void(Eigen::Index, Eigen::Index)> handleRange;
    handleRange = [=, &handleRange, &barrier, &f](Eigen::Index firstIdx,
                                                  Eigen::Index lastIdx) {
      if (mem_hint != nullptr) {
        // LOG(INFO) << std::to_string(block.size) << " " << std::to_string(lastIdx-firstIdx);
      }
      while (lastIdx - firstIdx > block.size) {
        // Split into halves and schedule the second half on a different thread.
        const Eigen::Index midIdx = firstIdx + Eigen::divup((lastIdx - firstIdx) / 2, block.size) * block.size;
        Schedule([=, &handleRange]() { handleRange(midIdx, lastIdx); });
        lastIdx = midIdx;
      }
      if (block.size > 10 && mem_hint != nullptr) {
        // LOG(INFO) << "Here"; 
        // int numa_node = port::NUMAGetMemAffinity(mem_hint + firstIdx);
        // int mem_hint_node = port::NUMAGetMemAffinity(mem_hint);
        // int last_idx_node = port::NUMAGetMemAffinity(mem_hint+lastIdx);
        // if (numa_node != mem_hint_node || numa_node != last_idx_node) {
          // LOG(INFO) << std::to_string(mem_hint_node) << " " << std::to_string(numa_node) << " " << std::to_string(last_idx_node) << " " << std::to_string(block.size);
        // }
        // int numa_node = std::rand() % 2;
        // if (numa_node != port::kNUMANoAffinity)
          // port::NUMASetThreadNodeAffinity(numa_node);
      }
      // Single block or less, execute directly.
      f(firstIdx, lastIdx);
      barrier.DecrementCount();
    };

    if (block.count <= NumThreads()) {
      // Avoid a thread hop by running the root of the tree and one block on the
      // main thread.
      handleRange(0, n);
    } else {
      // Execute the root in the thread pool to avoid running work on more than
      // numThreads() threads.
      Schedule([=, &handleRange]() { handleRange(0, n); });
    }

    barrier.Wait();

}

// This functionality is similar to parallelFor, except that reasoning about
// the number of shards used is significantly easier.
void ThreadPool::ParallelForFixedBlockSizeScheduling(
    const int64 total, const int64 block_size,
    const std::function<void(int64, int64)>& fn) {
  const int num_shards_used =
      NumShardsUsedByFixedBlockSizeScheduling(total, block_size);
  if (num_shards_used == 1) {
    fn(0, total);
    return;
  }

  // Adapted from Eigen's parallelFor implementation.
  BlockingCounter counter(num_shards_used);
  std::function<void(int64, int64)> handle_range =
      [=, &handle_range, &counter, &fn](int64 first, int64 last) {
        while (last - first > block_size) {
          // Find something near the midpoint which is a multiple of block size.
          const int64 mid = first + ((last - first) / 2 + block_size - 1) /
                                        block_size * block_size;
          Schedule([=, &handle_range]() { handle_range(mid, last); });
          last = mid;
        }
        // Single block or less, execute directly.
        fn(first, last);
        counter.DecrementCount();  // The shard is done.
      };
  if (num_shards_used <= NumThreads()) {
    // Avoid a thread hop by running the root of the tree and one block on the
    // main thread.
    handle_range(0, total);
  } else {
    // Execute the root in the thread pool to avoid running work on more than
    // numThreads() threads.
    Schedule([=, &handle_range]() { handle_range(0, total); });
  }
  counter.Wait();
}

void ThreadPool::ParallelFor(int64 total, int64 cost_per_unit,
                             const std::function<void(int64, int64)>& fn) {
  ThreadPool::ParallelFor(total, cost_per_unit, fn, nullptr);
}

void ThreadPool::ParallelFor(int64 total, int64 cost_per_unit,
                             const std::function<void(int64, int64)>& fn, const void *mem_hint) {
  CHECK_GE(total, 0);
  CHECK_EQ(total, (int64)(Eigen::Index)total);

  ThreadPool::ParallelForNonFixedBlockSizeScheduling(
    total, Eigen::TensorOpCost(0, 0, cost_per_unit), nullptr,
    [&fn](Eigen::Index first, Eigen::Index last) { fn(first, last); }, mem_hint);

  // threadpool_device_->parallelFor(
  //     total, Eigen::TensorOpCost(0, 0, cost_per_unit),
  //     [&fn](Eigen::Index first, Eigen::Index last) { fn(first, last); });
    
  
}

// void ThreadPool::ParallelFor(int64 total, int64 cost_per_unit,
//                              const std::function<void(int64, int64)>& fn) {
//   CHECK_GE(total, 0);
//   CHECK_EQ(total, (int64)(Eigen::Index)total);
//   threadpool_device_->parallelFor(
//       total, Eigen::TensorOpCost(0, 0, cost_per_unit),
//       [&fn](Eigen::Index first, Eigen::Index last) { fn(first, last); });
  
// }

void ThreadPool::ParallelForWithWorkerId(
    int64 total, int64 cost_per_unit,
    const std::function<void(int64, int64, int)>& fn) {
  CHECK_GE(total, 0);
  CHECK_EQ(total, (int64)(Eigen::Index)total);

  threadpool_device_->parallelFor(total,
                                  Eigen::TensorOpCost(0, 0, cost_per_unit),
                                  [this, &fn](int64 start, int64 limit) {
                                    // ParallelFor may use the current thread to
                                    // do some work synchronously. When calling
                                    // CurrentThreadId() from outside of the
                                    // thread pool, we get -1, so we can shift
                                    // every id up by 1.
                                    int id = CurrentThreadId() + 1;
                                    fn(start, limit, id);
                                  });
}

void ThreadPool::ParallelForWithWorkerId(
    int64 total, const SchedulingParams& scheduling_params,
    const std::function<void(int64, int64, int)>& fn) {
  ParallelFor(total, scheduling_params, [this, &fn](int64 start, int64 limit) {
    // We may use the current thread to do some work synchronously.
    // When calling CurrentThreadId() from outside of the thread
    // pool, we get -1, so we can shift every id up by 1.
    int id = CurrentThreadId() + 1;
    fn(start, limit, id);
  });
}

int ThreadPool::NumThreads() const {
  return underlying_threadpool_->NumThreads();
}

int ThreadPool::CurrentThreadId() const {
  return underlying_threadpool_->CurrentThreadId();
}

void ThreadPool::ScheduleWithHint(std::function<void()> fn, int start,
                                  int limit) {
  underlying_threadpool_->ScheduleWithHint(std::move(fn), start, limit);
}

void ThreadPool::SetStealPartitions(
    const std::vector<std::pair<unsigned, unsigned>>& partitions) {
  // ThreadPool::SetStealPartitions is only called in the constructor of
  // RunHandlerPool::Impl, which currently instantiates ThreadPool using a
  // constructor that does not take user_threadpool. Thus we assume
  // eigen_threadpool_ is not null here.
  DCHECK(eigen_threadpool_ != nullptr);
  eigen_threadpool_->SetStealPartitions(partitions);
}

Eigen::ThreadPoolInterface* ThreadPool::AsEigenThreadPool() const {
  DCHECK(underlying_threadpool_ != nullptr);
  return underlying_threadpool_;
}
}  // namespace thread
}  // namespace tensorflow
