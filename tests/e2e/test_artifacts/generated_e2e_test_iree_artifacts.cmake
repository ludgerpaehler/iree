iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-c36c63b0-220a-4d78-8ade-c45ce47d89d3"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32.tflite"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32.mlir"
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-c36c63b0-220a-4d78-8ade-c45ce47d89d3
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-0e466f69-91d6-4e50-b62b-a82b6213a231"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32.tflite"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32.mlir"
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-0e466f69-91d6-4e50-b62b-a82b6213a231
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-5afc3014-d29d-4e88-a840-fbaf678acf2b"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32.tflite"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32.mlir"
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-5afc3014-d29d-4e88-a840-fbaf678acf2b
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-cc69d69f-6d1f-4a1a-a31e-e021888d0d28"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32.tflite"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32.mlir"
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-cc69d69f-6d1f-4a1a-a31e-e021888d0d28
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-e3997104-a3d2-46b4-9fbf-39069906d123"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8.tflite"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8.mlir"
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-e3997104-a3d2-46b4-9fbf-39069906d123
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-73a0402e-271b-4aa8-a6a5-ac05839ca569"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_73a0402e-271b-4aa8-a6a5-ac05839ca569_MobileBertSquad_fp16.tflite"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_73a0402e-271b-4aa8-a6a5-ac05839ca569_MobileBertSquad_fp16.mlir"
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-73a0402e-271b-4aa8-a6a5-ac05839ca569
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-78eab9e5-9ff1-4769-9b55-933c81cc9a0f"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_78eab9e5-9ff1-4769-9b55-933c81cc9a0f_MobileNetV1_fp32.0_float.tflite"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_78eab9e5-9ff1-4769-9b55-933c81cc9a0f_MobileNetV1_fp32.mlir"
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-78eab9e5-9ff1-4769-9b55-933c81cc9a0f
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-7d45f8e5-bb5e-48d0-928d-8f125104578f"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32.0_224.tflite"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32.mlir"
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-7d45f8e5-bb5e-48d0-928d-8f125104578f
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-58855e40-eba9-4a71-b878-6b35e3460244"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32.tflite"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32.mlir"
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-58855e40-eba9-4a71-b878-6b35e3460244
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-bc1338be-e3df-44fd-82e4-40ba9560a073"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_bc1338be-e3df-44fd-82e4-40ba9560a073_PersonDetect_int8.tflite"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_bc1338be-e3df-44fd-82e4-40ba9560a073_PersonDetect_int8.mlir"
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-bc1338be-e3df-44fd-82e4-40ba9560a073
)

iree_import_tflite_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-4a6f545e-1b4e-41a5-9236-792aa578184b"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_4a6f545e-1b4e-41a5-9236-792aa578184b_EfficientNet_int8.tflite"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_4a6f545e-1b4e-41a5-9236-792aa578184b_EfficientNet_int8.mlir"
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-4a6f545e-1b4e-41a5-9236-792aa578184b
)

iree_import_tf_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-ecf5c970-ee97-49f0-a4ed-df1f34e9d493"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_ecf5c970-ee97-49f0-a4ed-df1f34e9d493_MiniLML12H384Uncased"
  ENTRY_FUNCTION
    "predict"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_ecf5c970-ee97-49f0-a4ed-df1f34e9d493_MiniLML12H384Uncased.mlir"
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-ecf5c970-ee97-49f0-a4ed-df1f34e9d493
)

iree_import_tf_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-39d157ad-f0ec-4a76-963b-d783beaed60f"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_39d157ad-f0ec-4a76-963b-d783beaed60f_BertForMaskedLMTF"
  ENTRY_FUNCTION
    "forward"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_39d157ad-f0ec-4a76-963b-d783beaed60f_BertForMaskedLMTF.mlir"
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-39d157ad-f0ec-4a76-963b-d783beaed60f
)

iree_import_tf_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-ebe7897f-5613-435b-a330-3cb967704e5e"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_ebe7897f-5613-435b-a330-3cb967704e5e_EfficientNetV2STF"
  ENTRY_FUNCTION
    "forward"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_ebe7897f-5613-435b-a330-3cb967704e5e_EfficientNetV2STF.mlir"
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-ebe7897f-5613-435b-a330-3cb967704e5e
)

iree_import_tf_model(
  TARGET_NAME
    "${PACKAGE_NAME}_iree-imported-model-c393b4fa-beb4-45d5-982a-c6328aa05d08"
  SOURCE
    "${ROOT_ARTIFACTS_DIR}/model_c393b4fa-beb4-45d5-982a-c6328aa05d08_Resnet50TF"
  ENTRY_FUNCTION
    "forward"
  OUTPUT_MLIR_FILE
    "${ROOT_ARTIFACTS_DIR}/iree_c393b4fa-beb4-45d5-982a-c6328aa05d08_Resnet50TF.mlir"
)

add_dependencies(iree-benchmark-import-models
  ${PACKAGE_NAME}_iree-imported-model-c393b4fa-beb4-45d5-982a-c6328aa05d08
)

iree_bytecode_module(
  NAME
    "iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32_e7e18b0f-c72d-4f1c-89b1-5afee70df6e9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9
)

iree_bytecode_module(
  NAME
    "iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32_e7e18b0f-c72d-4f1c-89b1-5afee70df6e9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9
)

iree_bytecode_module(
  NAME
    "iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32_e7e18b0f-c72d-4f1c-89b1-5afee70df6e9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9
)

iree_bytecode_module(
  NAME
    "iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32_e7e18b0f-c72d-4f1c-89b1-5afee70df6e9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9
)

iree_bytecode_module(
  NAME
    "iree-module-e3997104-a3d2-46b4-9fbf-39069906d123-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8_e7e18b0f-c72d-4f1c-89b1-5afee70df6e9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-e3997104-a3d2-46b4-9fbf-39069906d123-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9
)

iree_bytecode_module(
  NAME
    "iree-module-73a0402e-271b-4aa8-a6a5-ac05839ca569-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_73a0402e-271b-4aa8-a6a5-ac05839ca569_MobileBertSquad_fp16.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_73a0402e-271b-4aa8-a6a5-ac05839ca569_MobileBertSquad_fp16_e7e18b0f-c72d-4f1c-89b1-5afee70df6e9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-73a0402e-271b-4aa8-a6a5-ac05839ca569-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9
)

iree_bytecode_module(
  NAME
    "iree-module-78eab9e5-9ff1-4769-9b55-933c81cc9a0f-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_78eab9e5-9ff1-4769-9b55-933c81cc9a0f_MobileNetV1_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_78eab9e5-9ff1-4769-9b55-933c81cc9a0f_MobileNetV1_fp32_e7e18b0f-c72d-4f1c-89b1-5afee70df6e9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-78eab9e5-9ff1-4769-9b55-933c81cc9a0f-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9
)

iree_bytecode_module(
  NAME
    "iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32_e7e18b0f-c72d-4f1c-89b1-5afee70df6e9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9
)

iree_bytecode_module(
  NAME
    "iree-module-58855e40-eba9-4a71-b878-6b35e3460244-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32_e7e18b0f-c72d-4f1c-89b1-5afee70df6e9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-58855e40-eba9-4a71-b878-6b35e3460244-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9
)

iree_bytecode_module(
  NAME
    "iree-module-bc1338be-e3df-44fd-82e4-40ba9560a073-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_bc1338be-e3df-44fd-82e4-40ba9560a073_PersonDetect_int8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_bc1338be-e3df-44fd-82e4-40ba9560a073_PersonDetect_int8_e7e18b0f-c72d-4f1c-89b1-5afee70df6e9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-bc1338be-e3df-44fd-82e4-40ba9560a073-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9
)

iree_bytecode_module(
  NAME
    "iree-module-4a6f545e-1b4e-41a5-9236-792aa578184b-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_4a6f545e-1b4e-41a5-9236-792aa578184b_EfficientNet_int8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_4a6f545e-1b4e-41a5-9236-792aa578184b_EfficientNet_int8_e7e18b0f-c72d-4f1c-89b1-5afee70df6e9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-4a6f545e-1b4e-41a5-9236-792aa578184b-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9
)

iree_bytecode_module(
  NAME
    "iree-module-ecf5c970-ee97-49f0-a4ed-df1f34e9d493-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_ecf5c970-ee97-49f0-a4ed-df1f34e9d493_MiniLML12H384Uncased.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_ecf5c970-ee97-49f0-a4ed-df1f34e9d493_MiniLML12H384Uncased_e7e18b0f-c72d-4f1c-89b1-5afee70df6e9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-ecf5c970-ee97-49f0-a4ed-df1f34e9d493-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9
)

iree_bytecode_module(
  NAME
    "iree-module-39d157ad-f0ec-4a76-963b-d783beaed60f-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_39d157ad-f0ec-4a76-963b-d783beaed60f_BertForMaskedLMTF.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_39d157ad-f0ec-4a76-963b-d783beaed60f_BertForMaskedLMTF_e7e18b0f-c72d-4f1c-89b1-5afee70df6e9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-39d157ad-f0ec-4a76-963b-d783beaed60f-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9
)

iree_bytecode_module(
  NAME
    "iree-module-ebe7897f-5613-435b-a330-3cb967704e5e-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_ebe7897f-5613-435b-a330-3cb967704e5e_EfficientNetV2STF.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_ebe7897f-5613-435b-a330-3cb967704e5e_EfficientNetV2STF_e7e18b0f-c72d-4f1c-89b1-5afee70df6e9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-ebe7897f-5613-435b-a330-3cb967704e5e-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9
)

iree_bytecode_module(
  NAME
    "iree-module-c393b4fa-beb4-45d5-982a-c6328aa05d08-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_c393b4fa-beb4-45d5-982a-c6328aa05d08_Resnet50TF.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_c393b4fa-beb4-45d5-982a-c6328aa05d08_Resnet50TF_e7e18b0f-c72d-4f1c-89b1-5afee70df6e9/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-c393b4fa-beb4-45d5-982a-c6328aa05d08-e7e18b0f-c72d-4f1c-89b1-5afee70df6e9
)

iree_bytecode_module(
  NAME
    "iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-6d0d5716-5525-44ad-b71d-8075ee1583a6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32_6d0d5716-5525-44ad-b71d-8075ee1583a6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-6d0d5716-5525-44ad-b71d-8075ee1583a6
)

iree_bytecode_module(
  NAME
    "iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-6d0d5716-5525-44ad-b71d-8075ee1583a6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32_6d0d5716-5525-44ad-b71d-8075ee1583a6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-6d0d5716-5525-44ad-b71d-8075ee1583a6
)

iree_bytecode_module(
  NAME
    "iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-6d0d5716-5525-44ad-b71d-8075ee1583a6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32_6d0d5716-5525-44ad-b71d-8075ee1583a6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-6d0d5716-5525-44ad-b71d-8075ee1583a6
)

iree_bytecode_module(
  NAME
    "iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-6d0d5716-5525-44ad-b71d-8075ee1583a6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32_6d0d5716-5525-44ad-b71d-8075ee1583a6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-6d0d5716-5525-44ad-b71d-8075ee1583a6
)

iree_bytecode_module(
  NAME
    "iree-module-e3997104-a3d2-46b4-9fbf-39069906d123-6d0d5716-5525-44ad-b71d-8075ee1583a6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8_6d0d5716-5525-44ad-b71d-8075ee1583a6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-e3997104-a3d2-46b4-9fbf-39069906d123-6d0d5716-5525-44ad-b71d-8075ee1583a6
)

iree_bytecode_module(
  NAME
    "iree-module-73a0402e-271b-4aa8-a6a5-ac05839ca569-6d0d5716-5525-44ad-b71d-8075ee1583a6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_73a0402e-271b-4aa8-a6a5-ac05839ca569_MobileBertSquad_fp16.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_73a0402e-271b-4aa8-a6a5-ac05839ca569_MobileBertSquad_fp16_6d0d5716-5525-44ad-b71d-8075ee1583a6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-73a0402e-271b-4aa8-a6a5-ac05839ca569-6d0d5716-5525-44ad-b71d-8075ee1583a6
)

iree_bytecode_module(
  NAME
    "iree-module-78eab9e5-9ff1-4769-9b55-933c81cc9a0f-6d0d5716-5525-44ad-b71d-8075ee1583a6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_78eab9e5-9ff1-4769-9b55-933c81cc9a0f_MobileNetV1_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_78eab9e5-9ff1-4769-9b55-933c81cc9a0f_MobileNetV1_fp32_6d0d5716-5525-44ad-b71d-8075ee1583a6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-78eab9e5-9ff1-4769-9b55-933c81cc9a0f-6d0d5716-5525-44ad-b71d-8075ee1583a6
)

iree_bytecode_module(
  NAME
    "iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-6d0d5716-5525-44ad-b71d-8075ee1583a6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32_6d0d5716-5525-44ad-b71d-8075ee1583a6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-6d0d5716-5525-44ad-b71d-8075ee1583a6
)

iree_bytecode_module(
  NAME
    "iree-module-58855e40-eba9-4a71-b878-6b35e3460244-6d0d5716-5525-44ad-b71d-8075ee1583a6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32_6d0d5716-5525-44ad-b71d-8075ee1583a6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-58855e40-eba9-4a71-b878-6b35e3460244-6d0d5716-5525-44ad-b71d-8075ee1583a6
)

iree_bytecode_module(
  NAME
    "iree-module-bc1338be-e3df-44fd-82e4-40ba9560a073-6d0d5716-5525-44ad-b71d-8075ee1583a6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_bc1338be-e3df-44fd-82e4-40ba9560a073_PersonDetect_int8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_bc1338be-e3df-44fd-82e4-40ba9560a073_PersonDetect_int8_6d0d5716-5525-44ad-b71d-8075ee1583a6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-bc1338be-e3df-44fd-82e4-40ba9560a073-6d0d5716-5525-44ad-b71d-8075ee1583a6
)

iree_bytecode_module(
  NAME
    "iree-module-4a6f545e-1b4e-41a5-9236-792aa578184b-6d0d5716-5525-44ad-b71d-8075ee1583a6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_4a6f545e-1b4e-41a5-9236-792aa578184b_EfficientNet_int8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_4a6f545e-1b4e-41a5-9236-792aa578184b_EfficientNet_int8_6d0d5716-5525-44ad-b71d-8075ee1583a6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-4a6f545e-1b4e-41a5-9236-792aa578184b-6d0d5716-5525-44ad-b71d-8075ee1583a6
)

iree_bytecode_module(
  NAME
    "iree-module-ecf5c970-ee97-49f0-a4ed-df1f34e9d493-6d0d5716-5525-44ad-b71d-8075ee1583a6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_ecf5c970-ee97-49f0-a4ed-df1f34e9d493_MiniLML12H384Uncased.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_ecf5c970-ee97-49f0-a4ed-df1f34e9d493_MiniLML12H384Uncased_6d0d5716-5525-44ad-b71d-8075ee1583a6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-ecf5c970-ee97-49f0-a4ed-df1f34e9d493-6d0d5716-5525-44ad-b71d-8075ee1583a6
)

iree_bytecode_module(
  NAME
    "iree-module-39d157ad-f0ec-4a76-963b-d783beaed60f-6d0d5716-5525-44ad-b71d-8075ee1583a6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_39d157ad-f0ec-4a76-963b-d783beaed60f_BertForMaskedLMTF.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_39d157ad-f0ec-4a76-963b-d783beaed60f_BertForMaskedLMTF_6d0d5716-5525-44ad-b71d-8075ee1583a6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-39d157ad-f0ec-4a76-963b-d783beaed60f-6d0d5716-5525-44ad-b71d-8075ee1583a6
)

iree_bytecode_module(
  NAME
    "iree-module-ebe7897f-5613-435b-a330-3cb967704e5e-6d0d5716-5525-44ad-b71d-8075ee1583a6"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_ebe7897f-5613-435b-a330-3cb967704e5e_EfficientNetV2STF.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_ebe7897f-5613-435b-a330-3cb967704e5e_EfficientNetV2STF_6d0d5716-5525-44ad-b71d-8075ee1583a6/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=mhlo"
    "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
    "--iree-llvm-target-cpu=cascadelake"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-ebe7897f-5613-435b-a330-3cb967704e5e-6d0d5716-5525-44ad-b71d-8075ee1583a6
)

iree_bytecode_module(
  NAME
    "iree-module-ecf5c970-ee97-49f0-a4ed-df1f34e9d493-09cb5300-7f73-45cf-9f68-e114c77ca030"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_ecf5c970-ee97-49f0-a4ed-df1f34e9d493_MiniLML12H384Uncased.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_ecf5c970-ee97-49f0-a4ed-df1f34e9d493_MiniLML12H384Uncased_09cb5300-7f73-45cf-9f68-e114c77ca030/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=mhlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-ecf5c970-ee97-49f0-a4ed-df1f34e9d493-09cb5300-7f73-45cf-9f68-e114c77ca030
)

iree_bytecode_module(
  NAME
    "iree-module-39d157ad-f0ec-4a76-963b-d783beaed60f-09cb5300-7f73-45cf-9f68-e114c77ca030"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_39d157ad-f0ec-4a76-963b-d783beaed60f_BertForMaskedLMTF.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_39d157ad-f0ec-4a76-963b-d783beaed60f_BertForMaskedLMTF_09cb5300-7f73-45cf-9f68-e114c77ca030/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=mhlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-39d157ad-f0ec-4a76-963b-d783beaed60f-09cb5300-7f73-45cf-9f68-e114c77ca030
)

iree_bytecode_module(
  NAME
    "iree-module-ebe7897f-5613-435b-a330-3cb967704e5e-09cb5300-7f73-45cf-9f68-e114c77ca030"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_ebe7897f-5613-435b-a330-3cb967704e5e_EfficientNetV2STF.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_ebe7897f-5613-435b-a330-3cb967704e5e_EfficientNetV2STF_09cb5300-7f73-45cf-9f68-e114c77ca030/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=mhlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-ebe7897f-5613-435b-a330-3cb967704e5e-09cb5300-7f73-45cf-9f68-e114c77ca030
)

iree_bytecode_module(
  NAME
    "iree-module-c393b4fa-beb4-45d5-982a-c6328aa05d08-09cb5300-7f73-45cf-9f68-e114c77ca030"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_c393b4fa-beb4-45d5-982a-c6328aa05d08_Resnet50TF.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_c393b4fa-beb4-45d5-982a-c6328aa05d08_Resnet50TF_09cb5300-7f73-45cf-9f68-e114c77ca030/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=cuda"
    "--iree-input-type=mhlo"
    "--iree-hal-cuda-llvm-target-arch=sm_80"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-c393b4fa-beb4-45d5-982a-c6328aa05d08-09cb5300-7f73-45cf-9f68-e114c77ca030
)

iree_bytecode_module(
  NAME
    "iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-cdf579a9-5446-403b-a991-802a6c702e65"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32_cdf579a9-5446-403b-a991-802a6c702e65/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvm-target-cpu=generic-rv64"
    "--iree-llvm-target-abi=lp64d"
    "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+v"
    "--riscv-v-vector-bits-min=512"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-cdf579a9-5446-403b-a991-802a6c702e65
)

iree_bytecode_module(
  NAME
    "iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-cdf579a9-5446-403b-a991-802a6c702e65"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32_cdf579a9-5446-403b-a991-802a6c702e65/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvm-target-cpu=generic-rv64"
    "--iree-llvm-target-abi=lp64d"
    "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+v"
    "--riscv-v-vector-bits-min=512"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-cdf579a9-5446-403b-a991-802a6c702e65
)

iree_bytecode_module(
  NAME
    "iree-module-78eab9e5-9ff1-4769-9b55-933c81cc9a0f-cdf579a9-5446-403b-a991-802a6c702e65"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_78eab9e5-9ff1-4769-9b55-933c81cc9a0f_MobileNetV1_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_78eab9e5-9ff1-4769-9b55-933c81cc9a0f_MobileNetV1_fp32_cdf579a9-5446-403b-a991-802a6c702e65/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvm-target-cpu=generic-rv64"
    "--iree-llvm-target-abi=lp64d"
    "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+v"
    "--riscv-v-vector-bits-min=512"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-78eab9e5-9ff1-4769-9b55-933c81cc9a0f-cdf579a9-5446-403b-a991-802a6c702e65
)

iree_bytecode_module(
  NAME
    "iree-module-e3997104-a3d2-46b4-9fbf-39069906d123-cdf579a9-5446-403b-a991-802a6c702e65"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8_cdf579a9-5446-403b-a991-802a6c702e65/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvm-target-cpu=generic-rv64"
    "--iree-llvm-target-abi=lp64d"
    "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+v"
    "--riscv-v-vector-bits-min=512"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-e3997104-a3d2-46b4-9fbf-39069906d123-cdf579a9-5446-403b-a991-802a6c702e65
)

iree_bytecode_module(
  NAME
    "iree-module-bc1338be-e3df-44fd-82e4-40ba9560a073-cdf579a9-5446-403b-a991-802a6c702e65"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_bc1338be-e3df-44fd-82e4-40ba9560a073_PersonDetect_int8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_bc1338be-e3df-44fd-82e4-40ba9560a073_PersonDetect_int8_cdf579a9-5446-403b-a991-802a6c702e65/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvm-target-cpu=generic-rv64"
    "--iree-llvm-target-abi=lp64d"
    "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+v"
    "--riscv-v-vector-bits-min=512"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-bc1338be-e3df-44fd-82e4-40ba9560a073-cdf579a9-5446-403b-a991-802a6c702e65
)

iree_bytecode_module(
  NAME
    "iree-module-4a6f545e-1b4e-41a5-9236-792aa578184b-cdf579a9-5446-403b-a991-802a6c702e65"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_4a6f545e-1b4e-41a5-9236-792aa578184b_EfficientNet_int8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_4a6f545e-1b4e-41a5-9236-792aa578184b_EfficientNet_int8_cdf579a9-5446-403b-a991-802a6c702e65/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=riscv64-pc-linux-gnu"
    "--iree-llvm-target-cpu=generic-rv64"
    "--iree-llvm-target-abi=lp64d"
    "--iree-llvm-target-cpu-features=+m,+a,+f,+d,+v"
    "--riscv-v-vector-bits-min=512"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-4a6f545e-1b4e-41a5-9236-792aa578184b-cdf579a9-5446-403b-a991-802a6c702e65
)

iree_bytecode_module(
  NAME
    "iree-module-bc1338be-e3df-44fd-82e4-40ba9560a073-6d9ce240-ec14-4d8f-a8e4-1b20aa17b4e4"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_bc1338be-e3df-44fd-82e4-40ba9560a073_PersonDetect_int8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_bc1338be-e3df-44fd-82e4-40ba9560a073_PersonDetect_int8_6d9ce240-ec14-4d8f-a8e4-1b20aa17b4e4/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=riscv32-pc-linux-gnu"
    "--iree-llvm-target-cpu=generic-rv32"
    "--iree-llvm-target-abi=ilp32"
    "--iree-llvm-target-cpu-features=+m,+a,+f,+zvl512b,+zve32x"
    "--riscv-v-vector-bits-min=512"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-bc1338be-e3df-44fd-82e4-40ba9560a073-6d9ce240-ec14-4d8f-a8e4-1b20aa17b4e4
)

iree_bytecode_module(
  NAME
    "iree-module-4a6f545e-1b4e-41a5-9236-792aa578184b-6d9ce240-ec14-4d8f-a8e4-1b20aa17b4e4"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_4a6f545e-1b4e-41a5-9236-792aa578184b_EfficientNet_int8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_4a6f545e-1b4e-41a5-9236-792aa578184b_EfficientNet_int8_6d9ce240-ec14-4d8f-a8e4-1b20aa17b4e4/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=riscv32-pc-linux-gnu"
    "--iree-llvm-target-cpu=generic-rv32"
    "--iree-llvm-target-abi=ilp32"
    "--iree-llvm-target-cpu-features=+m,+a,+f,+zvl512b,+zve32x"
    "--riscv-v-vector-bits-min=512"
    "--riscv-v-fixed-length-vector-lmul-max=8"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-4a6f545e-1b4e-41a5-9236-792aa578184b-6d9ce240-ec14-4d8f-a8e4-1b20aa17b4e4
)

iree_bytecode_module(
  NAME
    "iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-1f2adf49-282e-4aff-9d4f-e63b1621f1e8"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32_1f2adf49-282e-4aff-9d4f-e63b1621f1e8/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-1f2adf49-282e-4aff-9d4f-e63b1621f1e8
)

iree_bytecode_module(
  NAME
    "iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-1f2adf49-282e-4aff-9d4f-e63b1621f1e8"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32_1f2adf49-282e-4aff-9d4f-e63b1621f1e8/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-1f2adf49-282e-4aff-9d4f-e63b1621f1e8
)

iree_bytecode_module(
  NAME
    "iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-1f2adf49-282e-4aff-9d4f-e63b1621f1e8"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32_1f2adf49-282e-4aff-9d4f-e63b1621f1e8/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-1f2adf49-282e-4aff-9d4f-e63b1621f1e8
)

iree_bytecode_module(
  NAME
    "iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-1f2adf49-282e-4aff-9d4f-e63b1621f1e8"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32_1f2adf49-282e-4aff-9d4f-e63b1621f1e8/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-1f2adf49-282e-4aff-9d4f-e63b1621f1e8
)

iree_bytecode_module(
  NAME
    "iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-1f2adf49-282e-4aff-9d4f-e63b1621f1e8"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32_1f2adf49-282e-4aff-9d4f-e63b1621f1e8/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-1f2adf49-282e-4aff-9d4f-e63b1621f1e8
)

iree_bytecode_module(
  NAME
    "iree-module-58855e40-eba9-4a71-b878-6b35e3460244-1f2adf49-282e-4aff-9d4f-e63b1621f1e8"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32_1f2adf49-282e-4aff-9d4f-e63b1621f1e8/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-58855e40-eba9-4a71-b878-6b35e3460244-1f2adf49-282e-4aff-9d4f-e63b1621f1e8
)

iree_bytecode_module(
  NAME
    "iree-module-e3997104-a3d2-46b4-9fbf-39069906d123-1f2adf49-282e-4aff-9d4f-e63b1621f1e8"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8_1f2adf49-282e-4aff-9d4f-e63b1621f1e8/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-e3997104-a3d2-46b4-9fbf-39069906d123-1f2adf49-282e-4aff-9d4f-e63b1621f1e8
)

iree_bytecode_module(
  NAME
    "iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-d463322c-24e6-4685-85ca-d541b41a405f"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32_d463322c-24e6-4685-85ca-d541b41a405f/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-d463322c-24e6-4685-85ca-d541b41a405f
)

iree_bytecode_module(
  NAME
    "iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-d463322c-24e6-4685-85ca-d541b41a405f"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32_d463322c-24e6-4685-85ca-d541b41a405f/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-d463322c-24e6-4685-85ca-d541b41a405f
)

iree_bytecode_module(
  NAME
    "iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-d463322c-24e6-4685-85ca-d541b41a405f"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32_d463322c-24e6-4685-85ca-d541b41a405f/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-d463322c-24e6-4685-85ca-d541b41a405f
)

iree_bytecode_module(
  NAME
    "iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-d463322c-24e6-4685-85ca-d541b41a405f"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32_d463322c-24e6-4685-85ca-d541b41a405f/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-d463322c-24e6-4685-85ca-d541b41a405f
)

iree_bytecode_module(
  NAME
    "iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-d463322c-24e6-4685-85ca-d541b41a405f"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32_d463322c-24e6-4685-85ca-d541b41a405f/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-d463322c-24e6-4685-85ca-d541b41a405f
)

iree_bytecode_module(
  NAME
    "iree-module-58855e40-eba9-4a71-b878-6b35e3460244-d463322c-24e6-4685-85ca-d541b41a405f"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32_d463322c-24e6-4685-85ca-d541b41a405f/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-58855e40-eba9-4a71-b878-6b35e3460244-d463322c-24e6-4685-85ca-d541b41a405f
)

iree_bytecode_module(
  NAME
    "iree-module-e3997104-a3d2-46b4-9fbf-39069906d123-f672a6b9-99fc-47ce-8b1b-8e5f44a541a1"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8_f672a6b9-99fc-47ce-8b1b-8e5f44a541a1/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=llvm-cpu"
    "--iree-input-type=tosa"
    "--iree-llvm-target-triple=aarch64-none-linux-android29"
    "--iree-flow-enable-data-tiling"
    "--iree-llvm-target-cpu-features=+dotprod"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-llvmcpu-enable-pad-consumer-fusion"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-e3997104-a3d2-46b4-9fbf-39069906d123-f672a6b9-99fc-47ce-8b1b-8e5f44a541a1
)

iree_bytecode_module(
  NAME
    "iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-c7eea358-d8d2-4199-9d75-bb741c399b1b"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32_c7eea358-d8d2-4199-9d75-bb741c399b1b/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-linux-android31"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-c7eea358-d8d2-4199-9d75-bb741c399b1b
)

iree_bytecode_module(
  NAME
    "iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-c7eea358-d8d2-4199-9d75-bb741c399b1b"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32_c7eea358-d8d2-4199-9d75-bb741c399b1b/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-linux-android31"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-c7eea358-d8d2-4199-9d75-bb741c399b1b
)

iree_bytecode_module(
  NAME
    "iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-c7eea358-d8d2-4199-9d75-bb741c399b1b"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32_c7eea358-d8d2-4199-9d75-bb741c399b1b/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-linux-android31"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-c7eea358-d8d2-4199-9d75-bb741c399b1b
)

iree_bytecode_module(
  NAME
    "iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-c7eea358-d8d2-4199-9d75-bb741c399b1b"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32_c7eea358-d8d2-4199-9d75-bb741c399b1b/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-linux-android31"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-c7eea358-d8d2-4199-9d75-bb741c399b1b
)

iree_bytecode_module(
  NAME
    "iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-c7eea358-d8d2-4199-9d75-bb741c399b1b"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32_c7eea358-d8d2-4199-9d75-bb741c399b1b/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-linux-android31"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-c7eea358-d8d2-4199-9d75-bb741c399b1b
)

iree_bytecode_module(
  NAME
    "iree-module-58855e40-eba9-4a71-b878-6b35e3460244-c7eea358-d8d2-4199-9d75-bb741c399b1b"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32_c7eea358-d8d2-4199-9d75-bb741c399b1b/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-linux-android31"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-58855e40-eba9-4a71-b878-6b35e3460244-c7eea358-d8d2-4199-9d75-bb741c399b1b
)

iree_bytecode_module(
  NAME
    "iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-d3038b95-c889-456a-bff6-5cbabd10f1ad"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32_d3038b95-c889-456a-bff6-5cbabd10f1ad/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-d3038b95-c889-456a-bff6-5cbabd10f1ad
)

iree_bytecode_module(
  NAME
    "iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-d3038b95-c889-456a-bff6-5cbabd10f1ad"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32_d3038b95-c889-456a-bff6-5cbabd10f1ad/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-d3038b95-c889-456a-bff6-5cbabd10f1ad
)

iree_bytecode_module(
  NAME
    "iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-d3038b95-c889-456a-bff6-5cbabd10f1ad"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32_d3038b95-c889-456a-bff6-5cbabd10f1ad/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-d3038b95-c889-456a-bff6-5cbabd10f1ad
)

iree_bytecode_module(
  NAME
    "iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-d3038b95-c889-456a-bff6-5cbabd10f1ad"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32_d3038b95-c889-456a-bff6-5cbabd10f1ad/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-d3038b95-c889-456a-bff6-5cbabd10f1ad
)

iree_bytecode_module(
  NAME
    "iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-d3038b95-c889-456a-bff6-5cbabd10f1ad"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32_d3038b95-c889-456a-bff6-5cbabd10f1ad/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-d3038b95-c889-456a-bff6-5cbabd10f1ad
)

iree_bytecode_module(
  NAME
    "iree-module-58855e40-eba9-4a71-b878-6b35e3460244-d3038b95-c889-456a-bff6-5cbabd10f1ad"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32_d3038b95-c889-456a-bff6-5cbabd10f1ad/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-58855e40-eba9-4a71-b878-6b35e3460244-d3038b95-c889-456a-bff6-5cbabd10f1ad
)

iree_bytecode_module(
  NAME
    "iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-70b823ca-2807-4531-8c00-e02af7d70466"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32_70b823ca-2807-4531-8c00-e02af7d70466/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=16"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-70b823ca-2807-4531-8c00-e02af7d70466
)

iree_bytecode_module(
  NAME
    "iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-70b823ca-2807-4531-8c00-e02af7d70466"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32_70b823ca-2807-4531-8c00-e02af7d70466/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=16"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-70b823ca-2807-4531-8c00-e02af7d70466
)

iree_bytecode_module(
  NAME
    "iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-70b823ca-2807-4531-8c00-e02af7d70466"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32_70b823ca-2807-4531-8c00-e02af7d70466/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=16"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-70b823ca-2807-4531-8c00-e02af7d70466
)

iree_bytecode_module(
  NAME
    "iree-module-58855e40-eba9-4a71-b878-6b35e3460244-70b823ca-2807-4531-8c00-e02af7d70466"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32_70b823ca-2807-4531-8c00-e02af7d70466/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=adreno-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=16"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-58855e40-eba9-4a71-b878-6b35e3460244-70b823ca-2807-4531-8c00-e02af7d70466
)

iree_bytecode_module(
  NAME
    "iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-8da35f2b-a042-4b7d-9dcf-5ebbc1728765"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32_8da35f2b-a042-4b7d-9dcf-5ebbc1728765/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-8da35f2b-a042-4b7d-9dcf-5ebbc1728765
)

iree_bytecode_module(
  NAME
    "iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-8da35f2b-a042-4b7d-9dcf-5ebbc1728765"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32_8da35f2b-a042-4b7d-9dcf-5ebbc1728765/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-8da35f2b-a042-4b7d-9dcf-5ebbc1728765
)

iree_bytecode_module(
  NAME
    "iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-8da35f2b-a042-4b7d-9dcf-5ebbc1728765"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32_8da35f2b-a042-4b7d-9dcf-5ebbc1728765/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-8da35f2b-a042-4b7d-9dcf-5ebbc1728765
)

iree_bytecode_module(
  NAME
    "iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-8da35f2b-a042-4b7d-9dcf-5ebbc1728765"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32_8da35f2b-a042-4b7d-9dcf-5ebbc1728765/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-8da35f2b-a042-4b7d-9dcf-5ebbc1728765
)

iree_bytecode_module(
  NAME
    "iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-8da35f2b-a042-4b7d-9dcf-5ebbc1728765"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32_8da35f2b-a042-4b7d-9dcf-5ebbc1728765/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-8da35f2b-a042-4b7d-9dcf-5ebbc1728765
)

iree_bytecode_module(
  NAME
    "iree-module-58855e40-eba9-4a71-b878-6b35e3460244-8da35f2b-a042-4b7d-9dcf-5ebbc1728765"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32_8da35f2b-a042-4b7d-9dcf-5ebbc1728765/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-58855e40-eba9-4a71-b878-6b35e3460244-8da35f2b-a042-4b7d-9dcf-5ebbc1728765
)

iree_bytecode_module(
  NAME
    "iree-module-73a0402e-271b-4aa8-a6a5-ac05839ca569-8da35f2b-a042-4b7d-9dcf-5ebbc1728765_demote_f32_to_16"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_73a0402e-271b-4aa8-a6a5-ac05839ca569_MobileBertSquad_fp16.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_73a0402e-271b-4aa8-a6a5-ac05839ca569_MobileBertSquad_fp16_8da35f2b-a042-4b7d-9dcf-5ebbc1728765_demote_f32_to_16/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-demote-f32-to-f16"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-73a0402e-271b-4aa8-a6a5-ac05839ca569-8da35f2b-a042-4b7d-9dcf-5ebbc1728765_demote_f32_to_16
)

iree_bytecode_module(
  NAME
    "iree-module-e3997104-a3d2-46b4-9fbf-39069906d123-8da35f2b-a042-4b7d-9dcf-5ebbc1728765_demote_f32_to_16"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8_8da35f2b-a042-4b7d-9dcf-5ebbc1728765_demote_f32_to_16/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-demote-f32-to-f16"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-e3997104-a3d2-46b4-9fbf-39069906d123-8da35f2b-a042-4b7d-9dcf-5ebbc1728765_demote_f32_to_16
)

iree_bytecode_module(
  NAME
    "iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-32a56c8d-cc6c-41b8-8620-1f8eda0b8223"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32_32a56c8d-cc6c-41b8-8620-1f8eda0b8223/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-32a56c8d-cc6c-41b8-8620-1f8eda0b8223
)

iree_bytecode_module(
  NAME
    "iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-32a56c8d-cc6c-41b8-8620-1f8eda0b8223"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32_32a56c8d-cc6c-41b8-8620-1f8eda0b8223/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-32a56c8d-cc6c-41b8-8620-1f8eda0b8223
)

iree_bytecode_module(
  NAME
    "iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-32a56c8d-cc6c-41b8-8620-1f8eda0b8223"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32_32a56c8d-cc6c-41b8-8620-1f8eda0b8223/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-32a56c8d-cc6c-41b8-8620-1f8eda0b8223
)

iree_bytecode_module(
  NAME
    "iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-32a56c8d-cc6c-41b8-8620-1f8eda0b8223"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32_32a56c8d-cc6c-41b8-8620-1f8eda0b8223/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-32a56c8d-cc6c-41b8-8620-1f8eda0b8223
)

iree_bytecode_module(
  NAME
    "iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-32a56c8d-cc6c-41b8-8620-1f8eda0b8223"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32_32a56c8d-cc6c-41b8-8620-1f8eda0b8223/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-32a56c8d-cc6c-41b8-8620-1f8eda0b8223
)

iree_bytecode_module(
  NAME
    "iree-module-58855e40-eba9-4a71-b878-6b35e3460244-32a56c8d-cc6c-41b8-8620-1f8eda0b8223"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32_32a56c8d-cc6c-41b8-8620-1f8eda0b8223/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-58855e40-eba9-4a71-b878-6b35e3460244-32a56c8d-cc6c-41b8-8620-1f8eda0b8223
)

iree_bytecode_module(
  NAME
    "iree-module-73a0402e-271b-4aa8-a6a5-ac05839ca569-32a56c8d-cc6c-41b8-8620-1f8eda0b8223_demote_f32_to_16"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_73a0402e-271b-4aa8-a6a5-ac05839ca569_MobileBertSquad_fp16.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_73a0402e-271b-4aa8-a6a5-ac05839ca569_MobileBertSquad_fp16_32a56c8d-cc6c-41b8-8620-1f8eda0b8223_demote_f32_to_16/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-flow-demote-f32-to-f16"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-73a0402e-271b-4aa8-a6a5-ac05839ca569-32a56c8d-cc6c-41b8-8620-1f8eda0b8223_demote_f32_to_16
)

iree_bytecode_module(
  NAME
    "iree-module-e3997104-a3d2-46b4-9fbf-39069906d123-32a56c8d-cc6c-41b8-8620-1f8eda0b8223_demote_f32_to_16"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8_32a56c8d-cc6c-41b8-8620-1f8eda0b8223_demote_f32_to_16/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-flow-demote-f32-to-f16"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-e3997104-a3d2-46b4-9fbf-39069906d123-32a56c8d-cc6c-41b8-8620-1f8eda0b8223_demote_f32_to_16
)

iree_bytecode_module(
  NAME
    "iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-6b601a8d-4824-42e0-bcc6-500c0c3fa346"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_c36c63b0-220a-4d78-8ade-c45ce47d89d3_DeepLabV3_fp32_6b601a8d-4824-42e0-bcc6-500c0c3fa346/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-c36c63b0-220a-4d78-8ade-c45ce47d89d3-6b601a8d-4824-42e0-bcc6-500c0c3fa346
)

iree_bytecode_module(
  NAME
    "iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-6b601a8d-4824-42e0-bcc6-500c0c3fa346"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_0e466f69-91d6-4e50-b62b-a82b6213a231_MobileSSD_fp32_6b601a8d-4824-42e0-bcc6-500c0c3fa346/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-0e466f69-91d6-4e50-b62b-a82b6213a231-6b601a8d-4824-42e0-bcc6-500c0c3fa346
)

iree_bytecode_module(
  NAME
    "iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-6b601a8d-4824-42e0-bcc6-500c0c3fa346"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_5afc3014-d29d-4e88-a840-fbaf678acf2b_PoseNet_fp32_6b601a8d-4824-42e0-bcc6-500c0c3fa346/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-5afc3014-d29d-4e88-a840-fbaf678acf2b-6b601a8d-4824-42e0-bcc6-500c0c3fa346
)

iree_bytecode_module(
  NAME
    "iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-6b601a8d-4824-42e0-bcc6-500c0c3fa346"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_cc69d69f-6d1f-4a1a-a31e-e021888d0d28_MobileBertSquad_fp32_6b601a8d-4824-42e0-bcc6-500c0c3fa346/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-cc69d69f-6d1f-4a1a-a31e-e021888d0d28-6b601a8d-4824-42e0-bcc6-500c0c3fa346
)

iree_bytecode_module(
  NAME
    "iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-6b601a8d-4824-42e0-bcc6-500c0c3fa346"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32_6b601a8d-4824-42e0-bcc6-500c0c3fa346/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-6b601a8d-4824-42e0-bcc6-500c0c3fa346
)

iree_bytecode_module(
  NAME
    "iree-module-58855e40-eba9-4a71-b878-6b35e3460244-6b601a8d-4824-42e0-bcc6-500c0c3fa346"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32_6b601a8d-4824-42e0-bcc6-500c0c3fa346/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-58855e40-eba9-4a71-b878-6b35e3460244-6b601a8d-4824-42e0-bcc6-500c0c3fa346
)

iree_bytecode_module(
  NAME
    "iree-module-73a0402e-271b-4aa8-a6a5-ac05839ca569-6b601a8d-4824-42e0-bcc6-500c0c3fa346_demote_f32_to_16"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_73a0402e-271b-4aa8-a6a5-ac05839ca569_MobileBertSquad_fp16.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_73a0402e-271b-4aa8-a6a5-ac05839ca569_MobileBertSquad_fp16_6b601a8d-4824-42e0-bcc6-500c0c3fa346_demote_f32_to_16/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
    "--iree-flow-demote-f32-to-f16"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-73a0402e-271b-4aa8-a6a5-ac05839ca569-6b601a8d-4824-42e0-bcc6-500c0c3fa346_demote_f32_to_16
)

iree_bytecode_module(
  NAME
    "iree-module-e3997104-a3d2-46b4-9fbf-39069906d123-6b601a8d-4824-42e0-bcc6-500c0c3fa346_demote_f32_to_16"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_e3997104-a3d2-46b4-9fbf-39069906d123_MobileBertSquad_int8_6b601a8d-4824-42e0-bcc6-500c0c3fa346_demote_f32_to_16/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vulkan-spirv"
    "--iree-input-type=tosa"
    "--iree-vulkan-target-triple=valhall-unknown-linux-android31"
    "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops"
    "--iree-hal-benchmark-dispatch-repeat-count=32"
    "--iree-flow-demote-f32-to-f16"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-e3997104-a3d2-46b4-9fbf-39069906d123-6b601a8d-4824-42e0-bcc6-500c0c3fa346_demote_f32_to_16
)

iree_bytecode_module(
  NAME
    "iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-75336abd-8108-462c-9ce3-15443e3f32f4"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_7d45f8e5-bb5e-48d0-928d-8f125104578f_MobileNetV2_fp32_75336abd-8108-462c-9ce3-15443e3f32f4/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vmvx"
    "--iree-input-type=tosa"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-7d45f8e5-bb5e-48d0-928d-8f125104578f-75336abd-8108-462c-9ce3-15443e3f32f4
)

iree_bytecode_module(
  NAME
    "iree-module-58855e40-eba9-4a71-b878-6b35e3460244-75336abd-8108-462c-9ce3-15443e3f32f4"
  SRC
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32.mlir"
  MODULE_FILE_NAME
    "${ROOT_ARTIFACTS_DIR}/iree_58855e40-eba9-4a71-b878-6b35e3460244_MobileNetV3Small_fp32_75336abd-8108-462c-9ce3-15443e3f32f4/module.vmfb"
  FLAGS
    "--iree-hal-target-backends=vmvx"
    "--iree-input-type=tosa"
  PUBLIC
)

add_dependencies(iree-benchmark-suites
  ${PACKAGE_NAME}_iree-module-58855e40-eba9-4a71-b878-6b35e3460244-75336abd-8108-462c-9ce3-15443e3f32f4
)