#!/bin/sh

#EXAMPLE: ./Experimentos.sh eliga 10 SETTINGS_eliga.yaml

ALGORITHM=$1
NRUNS=$2
FILE=$3

let N=$(expr length $FILE)-5
NAME=$(expr substr $FILE 1 $N)


current_time=$(date "+%Y.%m.%d-%H.%M.%S")

echo "--------------------------------------------" >> $ALGORITHM---$NAME.txt
echo "$current_time" >> $ALGORITHM---$NAME.txt
echo "--------------------------------------------" >> $ALGORITHM---$NAME.txt



echo " " >> $ALGORITHM---$NAME.txt
echo " " >> $ALGORITHM---$NAME.txt
echo "=======================================================" >> $ALGORITHM---$NAME.txt
echo "Ejecutando $NRUNS corridas para $ALGORITHM con $NAME"    >> $ALGORITHM---$NAME.txt
echo "=======================================================" >> $ALGORITHM---$NAME.txt
echo " "


for (( ii=1; ii<=$NRUNS; ii++ ))
do
  echo "Corrida $ii de $NRUNS..."
  
  echo "Corrida $ii de $NRUNS..."              >> $ALGORITHM---$NAME.txt
  python $ALGORITHM.py -settings $FILE         >> $ALGORITHM---$NAME.txt
  echo " "                                     >> $ALGORITHM---$NAME.txt

done

echo " "
echo "Ejecución finalizada con éxito!!"


echo " "                                >> $ALGORITHM---$NAME.txt
echo "Ejecución finalizada con éxito!!" >> $ALGORITHM---$NAME.txt

echo "########################################################################################################## " >> $ALGORITHM---$NAME.txt
