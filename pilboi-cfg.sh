#!/bin/bash

#C
echo -n -e \\x00\\x00 > ~/pilboi-cfg.bin
#lip
echo -n -e \\x01\\x00 >> ~/pilboi-cfg.bin
#vin
echo -n -e \\x00\\x00 >> ~/pilboi-cfg.bin
#ace
echo -n -e \\x00\\x00 >> ~/pilboi-cfg.bin
#air
echo -n -e \\x00\\x00 >> ~/pilboi-cfg.bin
#cen
echo -n -e \\x01\\x00 >> ~/pilboi-cfg.bin
#iron
echo -n -e \\x01\\x00 >> ~/pilboi-cfg.bin
#mag
echo -n -e \\x02\\x00 >> ~/pilboi-cfg.bin

#dd < /dev/zero bs=4080  count=1 >> ~/pilboi-cfg.bin
