for i in $(seq 0 49); do
    echo $i
    python ./script/MBAR.py --idx $i &    
done
