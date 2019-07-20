python run_xlnet_dream.py --data_dir=data --xlnet_model=xlnet-large-cased --output_dir=xlnet_dream --max_seq_length=512 --do_train --do_eval --train_batch_size=32 --eval_batch_size=1 --learning_rate=3e-5 --num_train_epochs=3 --warmup_proportion=0.1 --gradient_accumulation_steps=16 
# && /root/shutdown.sh
