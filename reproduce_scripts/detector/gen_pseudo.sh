mkdir A2_pseudo
mkdir A2_pseudo/{0..17}
python tools/get_rawframes_a2pseudo.py
for i in {0..17}
do
    echo "copy $i"
    cp A2_pseudo/$i/* /ssd3/data/ai-city-2022/Track3/raw_frames/combine_A1A2Pseudo_vidconv_round2_bg/$i/ -r
done
rm  A2_pseudo/* -rf
python build_file_list.py
echo "Finish !"
