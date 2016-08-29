package = "gvnn"
version = "scm-1"

source = {
   url = "git://github.com/ankurhanda/gvnn.git",
}

description = {
   summary = "gvnn: Neural Network Library for Geometric Vision",
   detailed = [[
   ]],
   homepage = "https://github.com/ankurhanda/gvnn",
   license = "GPL"
}

dependencies = {
   "torch >= 7.0",
   "cutorch >= 1.0",
   "nn >= 1.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}
