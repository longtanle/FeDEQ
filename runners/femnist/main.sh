python3 main.py \
--inner_mode epoch \
--num_rounds 150 \
--num_clients 100 \
--eval_every 1 \
--inner_epochs 5 \
--personalized_epochs 3 \
--batch_size 10  \
--learning_rate 0.01 \
--seed 100 \
--dataset femnist \
$@
