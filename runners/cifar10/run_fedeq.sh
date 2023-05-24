bash runners/cifar10/main.sh \
--trainer fedeq_resnet --model deq_resnet_m --rho 0.01 --lam_admm 0.01 \
$@
