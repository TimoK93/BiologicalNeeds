DATA="DIC-C2DH-HeLa"
SEQUENCE="01"

./infer_ctc_data/infer_ctc_data "../${DATA}/${SEQUENCE}" \
"./models/${DATA}/best_iou_model.pth" "./models/${DATA}/config.json" \
--shifts 57

mkdir ../mht_input_${DATA}_${SEQUENCE}
mkdir ../mht_input_${DATA}_${SEQUENCE}/${DATA}
cp  ../${DATA}/${SEQUENCE} ../mht_input_${DATA}_${SEQUENCE}/${DATA} -r
mv ../${DATA}/${SEQUENCE}_RES ../mht_input_${DATA}_${SEQUENCE}/${DATA}

./mht/inference --single-inference --dataset "${DATA}" --sequence \
  "${SEQUENCE}"\
  --data-root "../mht_input_${DATA}_${SEQUENCE}"\
  --destination-root "../mht_result_${DATA}_${SEQUENCE}" --subset challenge

./mht/interpolate --single-sequence --dataset "${DATA}" --sequence \
  "${SEQUENCE}"\
  --data-root "../mht_result_${DATA}_${SEQUENCE}"\
  --destination-root "../mht_result2_${DATA}_${SEQUENCE}"

./mht/postprocess --single-sequence --dataset "${DATA}" --sequence \
  "${SEQUENCE}"\
  --data-root "../mht_result2_${DATA}_${SEQUENCE}"

mv ../mht_result2_${DATA}_${SEQUENCE}/${DATA}/${SEQUENCE}_RES ../${DATA}
rm ../mht_input_${DATA}_${SEQUENCE} -R
rm ../mht_result_${DATA}_${SEQUENCE} -R
rm ../mht_result2_${DATA}_${SEQUENCE} -R
