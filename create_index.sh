python3 -m colbert.index_with_compression \
--amp --doc_maxlen 256 --mask-punctuation --bsize 64 \
--checkpoint /home/ubuntu/Conversational-QA/checkpoints/colbert.dnn \
--collection ./data/subsample/corpus.tsv \
--index_root index/ --index_name QRECC \
--root /home/ubuntu/Conversational-QA/experiments/ --experiment QRECC \
--partitions 8192 --sample 0.01 \
--compression_level 1 --compression_thresholds ./compression_thresholds.csv \
--nproc_per_node=1 --chunksize=1.0
