bash runners/cifar100/main.sh \
--trainer fedeq_resnet --model deq_resnet_m --learning_rate 0.01 --rho 0.01 --lam_admm 0.01 \
$@
