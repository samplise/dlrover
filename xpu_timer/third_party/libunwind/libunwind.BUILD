# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

LIBUNWIND_COPTS = [
    "-DHAVE_CONFIG_H",
    "-D_GNU_SOURCE",
    "-Iexternal/org_gnu_libunwind/include",
    "-Iexternal/org_gnu_libunwind/include/tdep",
    "-Iexternal/org_gnu_libunwind/src",
]

filegroup(
    name = "internal_headers",
    srcs = [
        "include/compiler.h",
        "include/config.h",
        "include/dwarf.h",
        "include/dwarf-eh.h",
        "include/dwarf_i.h",
        "include/libunwind.h",
        "include/libunwind-common.h",
        "include/libunwind-coredump.h",
        "include/libunwind-dynamic.h",
        "include/libunwind-x86_64.h",
        "include/libunwind_i.h",
        "include/mempool.h",
        "include/remote.h",
        "include/tdep-x86_64/dwarf-config.h",
        "include/tdep-x86_64/libunwind_i.h",
        "include/tdep/dwarf-config.h",
        "include/tdep/libunwind_i.h",
        "include/unwind.h",
        "src/elf32.h",
        "src/elf64.h",
        "src/elfxx.h",
        "src/os-linux.h",
        "src/x86_64/init.h",
        "src/x86_64/ucontext_i.h",
        "src/x86_64/unwind_i.h",
    ],
)

filegroup(
    name = "sources_common",
    srcs = [
        "src/dwarf/Gexpr.c",
        "src/dwarf/Gfde.c",
        "src/dwarf/Gfind_proc_info-lsb.c",
        "src/dwarf/Gfind_unwind_table.c",
        "src/dwarf/Gparser.c",
        "src/dwarf/Gpe.c",
        "src/dwarf/global.c",
        "src/elf64.h",
        "src/elfxx.c",
        "src/elfxx.h",
        "src/mi/Gaddress_validator.c",
        "src/mi/Gdestroy_addr_space.c",
        "src/mi/Gdyn-extract.c",
        "src/mi/Gdyn-remote.c",
        "src/mi/Gfind_dynamic_proc_info.c",
        "src/mi/Gget_accessors.c",
        "src/mi/Gget_proc_name.c",
        "src/mi/Gget_reg.c",
        "src/mi/Gput_dynamic_unwind_info.c",
        "src/mi/flush_cache.c",
        "src/mi/init.c",
        "src/mi/mempool.c",
        "src/os-linux.c",
        "src/x86_64/Gcreate_addr_space.c",
        "src/x86_64/Gglobal.c",
        "src/x86_64/Ginit.c",
        "src/x86_64/Ginit_local.c",
        "src/x86_64/Gos-linux.c",
        "src/x86_64/Gregs.c",
        "src/x86_64/Gresume.c",
        "src/x86_64/Gstash_frame.c",
        "src/x86_64/Gstep.c",
        "src/x86_64/getcontext.S",
        "src/x86_64/is_fpreg.c",
        "src/x86_64/setcontext.S",
        ":internal_headers",
    ],
)

cc_library(
    name = "unwind",
    srcs = [
        ":internal_headers",
        ":sources_common",
    ],
    hdrs = [
        "include/config.h",
        "include/libunwind.h",
        "include/libunwind-aarch64.h",
        "include/libunwind-common.h",
        "include/libunwind-dynamic.h",
        "include/libunwind-x86_64.h",
    ],
    copts = LIBUNWIND_COPTS + [
        # Assume our inferior doesn't have frame pointers, regardless of
        # whether we ourselves do or not.
        "-DNO_FRAME_POINTER",
        "-fno-common",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
