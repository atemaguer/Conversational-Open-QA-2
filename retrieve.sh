python3 -m colbert.retrieve \
--amp --doc_maxlen 256 --mask-punctuation --bsize 64 \
--queries /home/ubuntu/Conversational-QA/data/subsample/queries.tsv \
--nprobe 32 --partitions 8192 --faiss_depth 1024 \
--index_root /home/ubuntu/Conversational-QA/index/ --index_name QRECC \
--checkpoint /home/ubuntu/Conversational-QA/checkpoints/colbert.dnn \
--root /home/ubuntu/Conversational-QA/experiments/ --experiment QRECC \
--compression_level 1 --compression_thresholds ./compression_thresholds.csv
