MACRO(ETISSPlugin ProjName)
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC ETISS)
    INSTALL(TARGETS ${ProjName}
        RUNTIME DESTINATION lib/plugins
        LIBRARY DESTINATION lib/plugins
        ARCHIVE DESTINATION lib/plugins
    )
    INSTALL(CODE
        "FILE(APPEND
            \"${CMAKE_INSTALL_PREFIX}/lib/plugins/list.txt\"
            \"${ProjName},\${CMAKE_INSTALL_PREFIX}/lib/plugins,${ProjName}\\n\"
        )"
    )
ENDMACRO()

MACRO(ETISSPluginArchName ProjName ArchName)
    ETISSPlugin(${ProjName})
    RegisterJITFileArch(${ArchName})
ENDMACRO()

MACRO(ETISSPluginArch ProjName)
    ETISSPluginArchName(${ProjName} ${ProjName})
ENDMACRO()
