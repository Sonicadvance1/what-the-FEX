#!/bin/bash
echo "Watching for $1"
while true ; do while ! [ $(FEXpidof "$1") ] ; do sleep 1; done ; ./what-the-FEX `FEXpidof "$1"` ; done
