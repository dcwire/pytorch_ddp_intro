import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function

a = torch.tensor([1., 2., 3.])

print(torch.square(a))
print(a ** 2)
print(a * a)

def time_pytorch_function(func, input):
    # CUDA IS ASYNC so can't use python time module
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        func(input)

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

b = torch.randn(10000, 10000).cuda()

def square_2(a):
    return a * a

def square_3(a):
    return a ** 2

print(time_pytorch_function(torch.square, b))
print(time_pytorch_function(square_2, b))
print(time_pytorch_function(square_3, b))

activities = [ProfilerActivity.CPU]

if torch.cuda.is_available():
    activities += [ProfilerActivity.CUDA]

print("=============")
print("Profiling torch.square")
print("=============")


# Now profile each function using pytorch profiler
# with torch.profiler.profile(
#         activities=activities,
#         with_stack=True,
#         experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
#         profile_memory=True,
#     ) as prof:
#     torch.square(b)

# print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=100))
with torch.profiler.profile(
        activities=activities,
    ) as prof:
    torch.square(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


print("=============")
print("Profiling a * a")
print("=============")

with torch.profiler.profile(
        activities=activities,
    ) as prof:
    square_2(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


print("=============")
print("Profiling a ** 2")
print("=============")

with torch.profiler.profile(
        activities=activities
    ) as prof:
    square_3(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))