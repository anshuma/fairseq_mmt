#!/usr/bin/bash
set -e

_mask=$1
_image_feat=$2

# set device
gpu=7

model_root_dir=checkpoints

# set task
task=multi30k-en2de
mask_data=$_mask
image_feat=$_image_feat

who=test1	#test1, test2
random_image_translation=0 #1
length_penalty=0.8

# set tag
model_dir_tag=$image_feat/$image_feat-$mask_data
#model_dir_tag=$image_feat/$image_feat-$mask_data_notFusionTop

if [ $task == "multi30k-en2de" ]; then
	tgt_lang=de
	if [ $mask_data == "mask0" ]; then
	        data_dir=multi30k.en-de
	elif [ $mask_data == "mask1" ]; then
	        data_dir=multi30k.en-de.mask1
	elif [ $mask_data == "mask2" ]; then
	        data_dir=multi30k.en-de.mask2
	elif [ $mask_data == "mask3" ]; then
	        data_dir=multi30k.en-de.mask3
	elif [ $mask_data == "mask4" ]; then
	        data_dir=multi30k.en-de.mask4
	elif [ $mask_data == "maskc" ]; then
	        data_dir=multi30k.en-de.maskc
	elif [ $mask_data == "maskp" ]; then
	        data_dir=multi30k.en-de.maskp
	fi
elif [ $task == 'multi30k-en2fr' ]; then
	tgt_lang=fr
	if [ $mask_data == "mask0" ]; then
        	data_dir=multi30k.en-fr
	elif [ $mask_data == "mask1" ]; then
	        data_dir=multi30k.en-fr.mask1
	elif [ $mask_data == "mask2" ]; then
      		data_dir=multi30k.en-fr.mask2
	elif [ $mask_data == "mask3" ]; then
	        data_dir=multi30k.en-fr.mask3
	elif [ $mask_data == "mask4" ]; then
	        data_dir=multi30k.en-fr.mask4
	elif [ $mask_data == "maskc" ]; then
	        data_dir=multi30k.en-fr.maskc
	elif [ $mask_data == "maskp" ]; then
	        data_dir=multi30k.en-fr.maskp
	fi
fi

if [ $image_feat == "vit_tiny_patch16_384" ]; then
	image_feat_path=data/$image_feat
	image_feat_dim=192
elif [ $image_feat == "vit_small_patch16_384" ]; then
	image_feat_path=data/$image_feat
	image_feat_dim=384
elif [ $image_feat == "vit_base_patch16_384" ]; then
	image_feat_path=data/$image_feat
	image_feat_dim=768
elif [ $image_feat == "vit_large_patch16_384" ]; then
	image_feat_path=data/$image_feat
	image_feat_dim=1024
fi

#image_feat_path=data/VisualBert_blip_large
#image_feat_dim=768
#image_feat_path=data/VisualBert_blip_large_DE
#image_feat_path=data/VisualBert_blip_large_DE_finetuned_en
#image_feat_path=data/VisualBert_blip_large_DE_8June
#image_feat_path=data/VisualBert_blip_large_DE_8June_avgpool
#image_feat_path=data/VisualBert_blip_large_DE_8June_maxpool
#image_feat_path=data/VisualBert_blip_large_DE_8June_finetune
#image_feat_dim=768
# data set
ensemble=12
batch_size=128
beam=5
src_lang=en

#model_dir=$model_root_dir/$task/$model_dir_tag
#model_dir=checkpoints/multi30k-en2de/VisualBert_blip_large_fusion_top/
#model_dir=checkpoints/multi30k-en2de/
#model_dir=checkpoints/multi30k-en2de/vit_base_patch14_reg4_dinov2_notFusionTop
#model_dir=checkpoints/multi30k-en2de/VisualBert_blip_large_DE_fusion_top/
#model_dir=checkpoints/multi30k-en2de/VisualBert_blip_large_EN_fusion_top_finetune/
#model_dir=checkpoints/multi30k-en2de/VisualBert_blip_large_DE_finetuned_de/
#model_dir=checkpoints/multi30k-en2de/VisualBert_blip_large_DE_pretrained_TGTDE_TRANSDE_8June/
#model_dir=checkpoints/multi30k-en2de/VisualBert_blip_large_DE_pretrained_TGTDE_TRANSDE_8June_validDE_de/
#model_dir=checkpoints/multi30k-en2de/VisualBert_blip_large_DE_8June_avgpool/
#model_dir=checkpoints/multi30k-en2de/VisualBert_blip_large_DE_8June_maxpool/
#model_dir=checkpoints/multi30k-en2de/VisualBert_blip_large_DE_pretrained_TGTDE_TRANSDE_8June_validDE_de_fusiontop
#model_dir=checkpoints/multi30k-en2de/VisualBert_blip_large_DE_pretrained_TGTDE_TRANSDE_8June_validDE_de_finetune
#model_dir=checkpoints/multi30k-en2de/vit_base_patch16_384/vit_base_patch16_384-mask0/
model_dir=checkpoints/multi30k-en2de/vit_base_patch16_384/QSRCIMG_KTGT_MultiHeadAttention3/
checkpoint=checkpoint_best.pt

if [ -n "$ensemble" ]; then
        if [ ! -e "$model_dir/last$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
        fi
        checkpoint=last$ensemble.ensemble.pt
fi
#checkpoint=checkpoint.best_loss_3.77.pt
output=$model_dir/translation_$who.log
export CUDA_VISIBLE_DEVICES=$gpu


cmd="fairseq-generate data-bin/$data_dir 
  -s $src_lang -t $tgt_lang 
  --path $model_dir/$checkpoint 
  --gen-subset $who 
  --batch-size $batch_size --beam $beam --lenpen $length_penalty 
  --quiet --remove-bpe
  --task image_mmt
  --image-feat-path $image_feat_path --image-feat-dim $image_feat_dim
  --output $model_dir/hypo.txt" 

if [ $random_image_translation -eq 1 ]; then
cmd=${cmd}" --random-image-translation "
fi

cmd=${cmd}" | tee "${output}
eval $cmd

python3 rerank.py $model_dir/hypo.txt $model_dir/hypo.sorted

if [ $task == "multi30k-en2de" ] && [ $who == "test" ]; then
	ref=data/multi30k/test.2016.de
elif [ $task == "multi30k-en2de" ] && [ $who == "test1" ]; then
	ref=data/multi30k/test.2017.de
elif [ $task == "multi30k-en2de" ] && [ $who == "test2" ]; then
	ref=data/multi30k/test.coco.de

elif [ $task == "multi30k-en2fr" ] && [ $who == 'test' ]; then
	ref=data/multi30k/test.2016.fr
elif [ $task == "multi30k-en2fr" ] && [ $who == 'test1' ]; then
	ref=data/multi30k/test.2017.fr
elif [ $task == "multi30k-en2fr" ] && [ $who == 'test2' ]; then
	ref=data/multi30k/test.coco.fr
fi	

hypo=$model_dir/hypo.sorted
python3 meteor.py $hypo $ref > $model_dir/meteor_$who.log
cat $model_dir/meteor_$who.log

# cal gate, follow Revisit-MMT
#python3 scripts/visual_awareness.py --input $model_dir_tag/gated.txt 

# cal accurary
python3 cal_acc.py $hypo $who $task
