# Tensor operations

## :point_right: cd .list나 numpy.ndarray로부터 tensor를 만들 수 있다.
``` python
torch.tensor([1, 2, 3])
torch.tensor(np.array(1))
```

## :point_right: .to 메소드로 tensor의 dtype, shape, device 등을 바꿀 수 있다.
``` python
x.to(dtype=torch.int)
```

## :point_right: .shape 혹은 .size()로 shape을 확인할 수 있다.
``` python
x.shape
x.size()
```

## :point_right: .reshape()로 shape을 변경할 수 있다.
``` python
x = torch.tensor([1, 2, 3, 4, 5, 6])
y = x.reshape((2, 3))
```

## :point_right: .view()로 shape을 변경할 수 있다:exclamation:
``` python
x = torch.tensor([1, 2, 3, 4, 5, 6])
y = x.view((2, 3))
```
> ### :question: reshape와 동일한가? :x:
> .reshape()와 동일하게 shape을 바꿔주지만, .view()는 메모리를 공유한다.  
> 즉, 위 코드에서 y를 변경하면 x도 변경된다.  
> 말 그대로 view만 변경

## :point_right: device 설정할 때, 몇 번 기기를 사용할 것인지 특정할 수 있다.
``` python
device = "cuda:0"
device = "cuda:1"
```

## :point_right: gpu에 있던 tensor를 .cpu()를 이용해 cpu로 옮길 수 있다.
``` python
x = torch.tensor([1, 2, 3, 4, 5, 6])
x = x.to(device="cuda")
y = x.cpu()
```

## :point_right: cpu에 저장된 tensor에 .tolist(), .numpy()를 사용할 수 있다.
> cpu가 아닌 device에 저장된 tensor에 사용하면 Error  
> ```y.cpu().numpy()``` 이렇게 이용하도록 하자.

## :point_Right: .item()을 이용해 스칼라 값을 읽을 수 있다.
``` python
x = torch.tensor([1])
x.item()
```
> element가 여러 개인 경우 item()을 사용하면 Error

## :point_right: .transpose()를 이용해 두 차원을 서로 바꿀 수 있다.
``` python
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])  # shape == (2, 3)

y = x.transpose(0, 1)  # shape == (3, 2)
y = torch.transpose(x, 0, 1)
```

## :point_right: .permute()를 이용해서 차원의 순서를 직접 지정할 수 있다.
``` python
# x.shape == (28, 28, 128)

y = x.permute(2, 0, 1)
# y.shape == (128, 28, 28)
```

## :point_right: .squeeze()를 이용해서 빈 차원을 제거할 수 있다.
``` python
# x.shape == (2, 1, 2, 2, 2, 2, 1)

y = x.squeeze()
# y.shape == (2, 2, 2, 2, 2)

z = x.squeeze(dim=1)  # 원하는 dim 선택 가능
# y.shape == (2, 2, 2, 2, 2, 1)
```

## :point_right: .unsqueeze()를 이용해서 빈 차원을 추가할 수 있다.
``` python
# x.shape == (28, 28)

y = x.unsqueeze(dim=0)
# y.shape == (1, 28, 28)
```

## :point_right: .chunk()를 이용해서 tensor를 분할할 수 있다.
``` python
# x.shape == (3, 28, 28)

y = torch.chunk(x, chunks=3, dim=0)
# 결과값인 y는 tensor의 튜플
# y[0].shape == (1, 28, 28)
```

## :point_right: .cat()을 이용하여 tensor를 병합할 수 있다.
``` python
# x, y shape == 각각 (1, 28, 28)

z = torch.cat((x, y))
# z.shape == (2, 28, 28)

z = torch.cat((x, y), dim=1)
# z.shape == (1, 56, 28)
```

## :point_right: .stack()을 이용하여 tensor를 새로운 dim으로 합칠 수 있다.
``` python
# x, y shape == 각각 (3, 28, 28)

z = torch.stack((x, y))
# z.shape == (2, 3, 28, 28)
```

## :point_right: indexing 할 수 있다.
``` python
x[:, 2]
x[1, 2]
x[:, [1, 2, 3]]
...
```

## :point_right: max, min, argmax, argmin, mean 등을 사용할 수 있다.
``` python
torch.max(x)
torch.max(x, dim=0)  # 결과값 : values, indices

torch.argmax(x, dim=0)  # 결과값 : indices
```
