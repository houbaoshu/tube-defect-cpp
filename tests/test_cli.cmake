file(REMOVE_RECURSE "${OUTPUT_DIR}")
execute_process(
    COMMAND "${CLI}" "${IMAGE}" --output-dir "${OUTPUT_DIR}" --print-json
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "CLI failed (${result}): ${stderr}\n${stdout}")
endif()
foreach(path result.json annotated.png evidence/color_spot_mask.png)
    if(NOT EXISTS "${OUTPUT_DIR}/${path}")
        message(FATAL_ERROR "Missing CLI output: ${OUTPUT_DIR}/${path}")
    endif()
endforeach()
file(READ "${OUTPUT_DIR}/result.json" report)
string(FIND "${report}" "\"type\": \"color_spot\"" match)
if(match EQUAL -1)
    message(FATAL_ERROR "Unexpected CLI report: ${report}")
endif()
