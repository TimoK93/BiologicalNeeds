DATA="PhC-C2DL-PSC"
SEQUENCE="01"

./infer_ctc_data/infer_ctc_data "../${DATA}/${SEQUENCE}" \
"./models/${DATA}/best_iou_model.pth" "./models/${DATA}/config.json" \
--shifts 12

mkdir ../pmbm_input_${DATA}_${SEQUENCE}
mkdir ../pmbm_input_${DATA}_${SEQUENCE}/${DATA}
cp  ../${DATA}/${SEQUENCE} ../pmbm_input_${DATA}_${SEQUENCE}/${DATA} -r
mv ../${DATA}/${SEQUENCE}_RES ../pmbm_input_${DATA}_${SEQUENCE}/${DATA}

./pmbm/inference --single-inference --challenge "${DATA}" --sequence \
  "${SEQUENCE}"\
  --data-root "../pmbm_input_${DATA}_${SEQUENCE}"\
  --destination-root "../pmbm_result_${DATA}_${SEQUENCE}"

./pmbm/interpolate --single-sequence --challenge "${DATA}" --sequence \
  "${SEQUENCE}"\
  --data-root "../pmbm_result_${DATA}_${SEQUENCE}"\
  --destination-root "../pmbm_result2_${DATA}_${SEQUENCE}"

./pmbm/postprocess --single-sequence --challenge "${DATA}" --sequence \
  "${SEQUENCE}"\
  --data-root "../pmbm_result2_${DATA}_${SEQUENCE}"

mv ../pmbm_result2_${DATA}_${SEQUENCE}/${DATA}/${SEQUENCE}_RES ../${DATA}
rm ../pmbm_input_${DATA}_${SEQUENCE} -R
rm ../pmbm_result_${DATA}_${SEQUENCE} -R
rm ../pmbm_result2_${DATA}_${SEQUENCE} -R
