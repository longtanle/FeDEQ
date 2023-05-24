python3 main.py \
--inner_mode epoch \
--num_rounds 400 \
--num_clients 200 \
--eval_every 1 \
--inner_epochs 5 \
--personalized_epochs 3 \
--labels_per_client 16 \
--batch_size 16  \
--learning_rate 0.1 \
--seed 100 \
--dataset shakespeare \
$@
