#
#	Copyright 2018 Infineon Technologies AG
#
#	This file is part of ETISS tool, see <https://github.com/tum-ei-eda/etiss>
#
#	The initial version of this software has been created with the funding support by the German Federal
#	Ministry of Education and Research(BMBF) in the project EffektiV under grant 01IS13022.
#
#	Redistribution and use in source and binary forms, with or without modification, are permitted
#	provided that the following conditions are met:
#
#	1. Redistributions of source code must retain the above copyright notice, this list of conditions and
#	the following disclaimer.
#
# 	2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
#	and the following disclaimer in the documentation and / or other materials provided with the distribution.
#
# 	3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
# 	or promote products derived from this software without specific prior written permission.
#
#	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
#	WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#	PARTICULAR PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
#	DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO,
#	PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#	HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING
#	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#	POSSIBILITY OF SUCH DAMAGE.
#
#
#	Author: Chair of Electronic Design Automation, TUM
#
#	Version 0.1
#

project(SW_example)


install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/
		DESTINATION examples/SW
		FILES_MATCHING
    	PATTERN "*.h"
    	PATTERN "*.cpp"
		PATTERN "*.c"
		PATTERN "*.S"
		PATTERN "*.s"
		PATTERN "README*"
    	PATTERN "Makefile"
        PATTERN "*.cmake"
		PATTERN "*.ld"
    	PATTERN ".svn" EXCLUDE
		PATTERN "*.in" EXCLUDE
		PATTERN "CMakeLists.txt" EXCLUDE
)

# Setup OR1K Makefile
configure_file(
  or1k/Makefile.in
  "${PROJECT_BINARY_DIR}/tobeinstalled/or1k/Makefile"
  @ONLY
 )

install(FILES "${PROJECT_BINARY_DIR}/tobeinstalled/or1k/Makefile"
		DESTINATION examples/SW/or1k
)

install(FILES riscv/CMakeLists.txt riscv/memsegs.ini
    DESTINATION examples/SW/riscv
)
install(FILES riscv_cpp/CMakeLists.txt riscv/memsegs.ini
    DESTINATION examples/SW/riscv_cpp
)
install(FILES riscv_dhry/CMakeLists.txt
    DESTINATION examples/SW/riscv_dhry
)