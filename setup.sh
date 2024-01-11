git submodule init --recursive
git submodule update --recursive
git apply RulePatch.patch
cd NOREC4DNA && python3 setup.py install