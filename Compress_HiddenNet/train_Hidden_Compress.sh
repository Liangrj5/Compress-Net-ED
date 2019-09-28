CUDA_VISIBLE_DEVICES='0' python3 -u main.py  \
	--cfg configs/Hidden_Compress.yaml  \
	--bs 10  \
	--nw 4  \
	--name test_rj \
	--debug
