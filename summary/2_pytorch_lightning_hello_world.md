# Pytorch Lightning 기본

## :point_right: 기본 pytorch에 비교한 장점
> 어떤 모델을 사용하는지와 별개로 항상 반복되는 구조들이 많은데,  
> 반복되는 부분을 직접 구현하지 않도록 해준다.  
> 그로 인해, 정말 중요한 것에 집중할 수 있도록 돕는다.

## :point_right: 주요 클래스들
> - Dataset
> - LightningDataModule
> - LightningModule
> - Trainer

## :point_right: Dataset
> __len__과 __getitem__을 구현한다.  
> __getitem__으로 index에 해당하는 데이터를 반환한다.

## :point_right: LightningDataModule
> prepare_data, setup, train_dataloader, val_dataloader, test_dataloader 메소드를 구현한다.  
> 디렉토리나 외부에서 데이터셋을 받아오고, Dataset형태로 만들어서 DataLoader를 반환한다.  
> LightningDataModule을 이용하면 pytorch lightning이 알아서 병렬 처리 등을 할 수 있다.  
> Trainer에 넘겨줄 때도 pytorch lightning이 알아서 dataloader를 꺼내서 사용하게 된다.  

## :point_right: LightningModule
> forward, training_step, validation_step, test_step등의 메소드를 구현한다.  
> __init__의 인자로 hyper parameters를 받도록 한다.  
> __init__에서 self.save_hyperparameters()를 통해 hyper parameters를 저장해놓고 사용할 수 있다. (logger에도 남는다.)
>> 여러 hyper parameters로 바꿔가며 학습 후, 어떤 파라미터가 사용되었나 확인 가능
> 
> forward는 forward propagation을 구현한다.  
> training_step에서는 loss를 반환하도록 구현하고, 그 외 step에서는 평가 메트릭 등을 로깅할 수 있다.  
>> training_step에서 loss를 반환하게 되면 Trainer가 loss를 바탕으로 파라미터를 학습한다.

## :point_right: Trainer
> 위처럼 미리 정의된 함수들을 재정의하여 작성만 하면 Trainer가 알아서 fit과 test등을 수행해준다.  

## :point_right: pl.seed_everything()
> random한 것들의 seed를 설정해준다.  
> 재연할 때 동일한 결과가 나오도록 보장해줄 수 있음  
> 하지만 안 되는 일부도 있으니, 필요할 떄 찾아보자.

## :point_right: argument 전달
> [문서 링크](https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html)  
> ** 나중에는 Lightning CLI도 써보자  
> 아래와 같이 argument를 구분하면 좋다.  
> 1. Trainer args (accelerator, devices, num_nodes, ...)  
>> ```parser = Trainer.add_argparse_args(parser)```  
>> 위처럼 작성하면 자동으로 --accelerator, --devices 등을 인자로 사용 가능
> 2. Model specific args (layer_dim, num_layers, learning_rate, ...)  
>> self.save_hyperparameters()로 저장할 수도 있다.  
>> (저장하게 되면 self.hparams로 읽어올 수 있다.)  
>> Model에 아래 메소드를 구현한다.
>> ``` python
>> @staticmethod
>> def add_model_specific_args(parent_parser):
>>     parser = parent_parser.add_argument_group(model_name)
>>     parser.add_argument(....)
>>     return parent_parser
>> ```
>> 구현 후, parent_parser = Model.add_model_specific_args(parent_parser)로 argument 추가
> 3. Program args (data_path, ...)
>> parser.add_argument로 직접 추가
> 
> 이렇게 모두 구현하면, program을 실행할 때 인자로 넘겨줄 수 있다.  
> - trainer에 사용 : ```trainer = Trainer.from_argparse_args(args)```  
> - model에 사용 : model의 **kwargs인자로 넘기기 위해, vars(args)로 dict형태로 args를 이용할 수 있다.  
