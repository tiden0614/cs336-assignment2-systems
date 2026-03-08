import torch

def test_accumulation():
    s = torch.tensor(0, dtype=torch.float32)

    def delta():
        a = s.type(torch.float32)
        return 10.000000 - a

    # Test 1: accu f32 into f32
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print(s, delta()) 

    # Test 2: accu f16 into f16
    s = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s, delta()) 

    # Test 3: accu f16 into f32
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s, delta()) 

    # Test 4: accu f16 into f32, but cast f16 to f32 before accu
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print(s, delta())
