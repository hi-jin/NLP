# Attention

## :point_right: sequence 2 sequence learning
> Input sequence (N) => Output sequence (M)  
> sequence의 관계에 따라 다양한 형태가 보인다.  
>> N->N  
>> N->1  
>> N->M  (M > 1)  
>
> 음성, 영상, 자연어 등  
> 데이터를 어떻게 바라볼 것인가에 따라 다양한 형태로 응용할 수 있다.  

## :point_right: Attention
> Sequence를 "**Blend**"  
> [링크](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)를 참고하자.  
> 관심있는 어떤 query Q에 대하여, Q와의 반응성을 바탕으로 입력데이터 X_n들을 Blend하자.  
>> softmax를 통과한 반응성(0~1)을 입력데이터에 곱하여 합쳐버리자.  
>
> 여기서 **반응성**을 어떻게 표현할 것인가?? (Q를 어떻게 잘 녹여낼 것인가?)  
>> 1. Q와 X_n 크기를 맞춰서 더한 후, Dense layer를 넣어서 반응성(중요도) 값을 얻는다. (additive)  
> 
>> 2. Q와 X_n를 concat하거나 크기를 적절히 맞추고 Dot product로 곱해버리자. (multiplicative)  
>
> 예시  
>> 지금까지의 결과를 바탕으로(Q), 입력 문자열(X)의 어떤 부분에 집중해서 번역할 것인가?  
>> 출력이 생성될 때마다 입력을 돌아보며 어울리는 입력을 찾는다.  
> 
> attention score로 나타나기 때문에, 설명 가능성을 가질 수 있다.  
