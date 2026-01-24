N=150000
L=150

command -v wgsim >/dev/null || mamba install -y -c bioconda wgsim

wgsim -N $N -1 $L -2 $L -e 0 -r 0 -R 0 -X 0 \
  reference/Homo_sapiens_assembly38.fasta \
  /tmp/control_ref_R1.fq /tmp/control_ref_R2.fq

awk 'NR%4==1{c++; $0="@A00685:52:DECOYFLOW:1:9999:9999:"sprintf("%05d",c)" 1:N:0:DECOYIDX+DECOYIDX"} {print}' \
  /tmp/control_ref_R1.fq | gzip -c > NA12877_R1.fq.gz

awk 'NR%4==1{c++; $0="@A00685:52:DECOYFLOW:1:9999:9999:"sprintf("%05d",c)" 2:N:0:DECOYIDX+DECOYIDX"} {print}' \
    /tmp/control_ref_R2.fq | gzip -c > NA12877_R2.fq.gz