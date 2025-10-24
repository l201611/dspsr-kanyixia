#!/usr/bin/env bash

echo ""
echo "###########"
echo "which dspsr"
which dspsr
echo "###########"
echo ""

export RESULTS_FOLDER=`date +"%Y-%m-%d-%H:%M:%S"`
export DS_BASE="/data/dspsr-test-data/functional-pipeline-tests"
export TEST_EXECUTION="$DS_BASE/$RESULTS_FOLDER"
mkdir -m 0755 $TEST_EXECUTION

export DS01="J0437-4715_MeerKAT_PTUSE_Analytic.dada"
export DS02=""
export DS03="J1644-4559_Parkes_CPSR2_Nyquist.cpsr2"
export DS04="J0835-4510_Parkes_UWL_Analytic.dada"
export DS05="J1939+2134_Parkes_UWL_Analytic.dada"
export DS06="G0024_1164211816_ch142_u.hdr"
export DS07="J1022+1001_R_Parkes_UWL_Analytic.dada"
export DS08="J0437-4715_MeerKAT_PTUSE_Coherence.sf"
export DS09="J1644-4559_MeerKAT_PTUSE_Coherence.sf"
export DS10="J0437-4715_MeerKAT_PTUSE_PPQQ.sf"
export DS11="J1644-4559_MeerKAT_PTUSE_PPQQ.sf"
export DS12="J0437-4715_MeerKAT_PTUSE_Intensity.sf"
export DS13="J1644-4559_MeerKAT_PTUSE_Intensity.sf"

export TESTCASE="$TEST_EXECUTION/TC01"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
echo TC01/DS01
dspsr $DS_BASE/DS01/$DS01 ; echo "dspsr $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
echo TC01/DS06
dspsr $DS_BASE/DS06/$DS06 ; echo "dspsr $DS_BASE/DS06/$DS06 exit code: $?" >> DS06.exit_code

export TESTCASE="$TEST_EXECUTION/TC02"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
echo TC02/DS01
dspsr -F 16384:D $DS_BASE/DS01/$DS01 ; echo "dspsr -F 16384:D $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
echo TC02/DS03
dspsr -F 64:D $DS_BASE/DS03/$DS03 ; echo "dspsr -F 64:D $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
echo TC02/DS04
dspsr -F 128:D $DS_BASE/DS04/$DS04 ; echo "dspsr -F 128:D $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
echo TC02/DS05
dspsr -F 128:D $DS_BASE/DS05/$DS05 ; echo "dspsr -F 128:D $DS_BASE/DS05/$DS05 exit code: $?" >> DS05.exit_code

export TESTCASE="$TEST_EXECUTION/TC03.1"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
echo TC03.1/DS01/a
dspsr -d 1 $DS_BASE/DS01/$DS01 ; echo "dspsr -d 1 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
echo TC03.1/DS01/b
dspsr -d 2 $DS_BASE/DS01/$DS01 ; echo "dspsr -d 2 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
echo TC03.1/DS01/c
dspsr -d 4 $DS_BASE/DS01/$DS01 ; echo "dspsr -d 4 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code

echo TC03.1/DS03/a
dspsr -d 1 -U 12545 $DS_BASE/DS03/$DS03 ; echo "dspsr -d 1 -U 12545 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
echo TC03.1/DS03/b
dspsr -d 2 -U 8448.02 $DS_BASE/DS03/$DS03 ; echo "dspsr -d 2 -U 8448.02 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
echo TC03.1/DS03/c
dspsr -d 4 -U 8448.02 $DS_BASE/DS03/$DS03 ; echo "dspsr -d 4 -U 8448.02 $DS_BASE/DS03/$DS03 exit code: $?"  >> DS03.exit_code

echo TC03.1/DS04/a
dspsr -d 1 -U 3329 $DS_BASE/DS04/$DS04 ; echo "dspsr -d 1 -U 3329 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
echo TC03.1/DS04/b
dspsr -d 2 -U 2305 $DS_BASE/DS04/$DS04 ; echo "dspsr -d 2 -U 2305 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
echo TC03.1/DS04/c
dspsr -d 4 -U 2305 $DS_BASE/DS04/$DS04 ; echo "dspsr -d 4 -U 2305 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code

export TESTCASE="$TEST_EXECUTION/TC03.2"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
echo TC03.2/DS01/a
dspsr -F 16384:D -d 1 $DS_BASE/DS01/$DS01 ; echo "dspsr -F 16384:D -d 1 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
echo TC03.2/DS01/b
dspsr -F 16384:D -d 2 $DS_BASE/DS01/$DS01 ; echo "dspsr -F 16384:D -d 2 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
echo TC03.2/DS01/c
dspsr -F 16384:D -d 4 $DS_BASE/DS01/$DS01 ; echo "dspsr -F 16384:D -d 4 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code

echo TC03.2/DS03/a
dspsr -F 128:D -d 1 $DS_BASE/DS03/$DS03 ; echo "dspsr -F 128:D -d 1 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
echo TC03.2/DS03/b
dspsr -F 128:D -d 2 $DS_BASE/DS03/$DS03 ; echo "dspsr -F 128:D -d 2 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
echo TC03.2/DS03/c
dspsr -F 128:D -d 4 $DS_BASE/DS03/$DS03 ; echo "dspsr -F 128:D -d 4 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code

echo TC03.2/DS04/a
dspsr -F 128:D -d 1 $DS_BASE/DS04/$DS04 ; echo "dspsr -F 128:D -d 1 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
echo TC03.2/DS04/b
dspsr -F 128:D -d 2 $DS_BASE/DS04/$DS04 ; echo "dspsr -F 128:D -d 2 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
echo TC03.2/DS04/c
dspsr -F 128:D -d 4 $DS_BASE/DS04/$DS04 ; echo "dspsr -F 128:D -d 4 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code

export TESTCASE="$TEST_EXECUTION/TC04.1"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
echo TC04.1/DS01/a
dspsr -b 64 $DS_BASE/DS01/$DS01 ; echo "dspsr -b 64 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
echo TC04.1/DS01/b
dspsr -b 1024 $DS_BASE/DS01/$DS01 ; echo "dspsr -b 1024 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
echo TC04.1/DS01/c
dspsr -b 4096 $DS_BASE/DS01/$DS01 ; echo "dspsr -b 4096 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code

echo TC04.1/DS03/a
dspsr -U 8449 -b 64 $DS_BASE/DS03/$DS03 ; echo "dspsr -U 8449 -b 64 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
echo TC04.1/DS03/b
dspsr -U 8449 -b 1024 $DS_BASE/DS03/$DS03 ; echo "dspsr -U 8449 -b 1024 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
echo TC04.1/DS03/c
dspsr -U 8449 -b 4096 $DS_BASE/DS03/$DS03 ; echo "dspsr -U 8449 -b 4096 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code

echo TC04.1/DS04/a
dspsr -U 2305 -b 64 $DS_BASE/DS04/$DS04 ; echo "dspsr -U 2305 -b 64 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
echo TC04.1/DS04/b
dspsr -U 2305 -b 1024 $DS_BASE/DS04/$DS04 ; echo "dspsr -U 2305 -b 1024 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
echo TC04.1/DS04/c
dspsr -U 2305 -b 4096 $DS_BASE/DS04/$DS04 ; echo "dspsr -U 2305 -b 4096 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code

export TESTCASE="$TEST_EXECUTION/TC04.2"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
echo TC04.2/DS01/a
dspsr -F 16384:D -b 64 $DS_BASE/DS01/$DS01; echo "dspsr -F 16384:D -b 64 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
echo TC04.2/DS01/b
dspsr -F 16384:D -b 1024 -U 265 $DS_BASE/DS01/$DS01; echo "dspsr -F 16384:D -b 1024 -U 265 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
echo TC04.2/DS01/c
dspsr -F 16384:D -b 4096 -U 1034 $DS_BASE/DS01/$DS01; echo "dspsr -F 16384:D -b 4096 -U 1034 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code

echo TC04.2/DS03/a
dspsr -F 128:D -b 64 $DS_BASE/DS03/$DS03; echo "dspsr -F 128:D -b 64 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
echo TC04.2/DS03/b
dspsr -F 128:D -b 1024 $DS_BASE/DS03/$DS03; echo "dspsr -F 128:D -b 1024 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
echo TC04.2/DS03/c
dspsr -F 128:D -b 4096 $DS_BASE/DS03/$DS03; echo "dspsr -F 128:D -b 4096 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code

echo TC04.2/DS04/a
dspsr -F 128:D -b 64 $DS_BASE/DS04/$DS04; echo "dspsr -F 128:D -b 64 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
echo TC04.2/DS04/b
dspsr -F 128:D -b 1024 $DS_BASE/DS04/$DS04; echo "dspsr -F 128:D -b 1024 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
echo TC04.2/DS04/c
dspsr -F 128:D -b 4096 $DS_BASE/DS04/$DS04; echo "dspsr -F 128:D -b 4096 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code

export TESTCASE="$TEST_EXECUTION/TC05.1"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
echo TC05.1/DS01/a
dspsr -L 1 $DS_BASE/DS01/$DS01 ; echo "dspsr -L 1 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
echo TC05.1/DS01/b
dspsr -L 8 $DS_BASE/DS01/$DS01 ; echo "dspsr -L 8 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
echo TC05.1/DS01/c
dspsr -L 30 $DS_BASE/DS01/$DS01 ; echo "dspsr -L 30 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code

echo TC05.1/DS03/a
dspsr -L 1 -U 8449 $DS_BASE/DS03/$DS03 ; echo "dspsr -L 1 -U 8449 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
echo TC05.1/DS03/b
dspsr -L 8 -U 8449 $DS_BASE/DS03/$DS03 ; echo "dspsr -L 8 -U 8449 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
echo TC05.1/DS03/c
dspsr -L 30 -U 8449 $DS_BASE/DS03/$DS03 ; echo "dspsr -L 30 -U 8449 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code

echo TC05.1/DS04/a
dspsr -L 1 -U 2305 $DS_BASE/DS04/$DS04 ; echo "dspsr -L 1 -U 2305 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
echo TC05.1/DS04/b
dspsr -L 8 -U 2305 $DS_BASE/DS04/$DS04 ; echo "dspsr -L 8 -U 2305 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
echo TC05.1/DS04/c
dspsr -L 30 -U 2305 $DS_BASE/DS04/$DS04 ; echo "dspsr -L 30 -U 2305 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code

export TESTCASE="$TEST_EXECUTION/TC05.2"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
echo TC05.2/DS01/a
dspsr -F 16384:D -L 1 $DS_BASE/DS01/$DS01 ; echo "dspsr -F 16384:D -L 1 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
echo TC05.2/DS01/b
dspsr -F 16384:D -L 8 $DS_BASE/DS01/$DS01 ; echo "dspsr -F 16384:D -L 8 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
echo TC05.2/DS01/c
dspsr -F 16384:D -L 30 $DS_BASE/DS01/$DS01 ; echo "dspsr -F 16384:D -L 30 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code

echo TC05.2/DS03/a
dspsr -F 128:D -L 1 $DS_BASE/DS03/$DS03 ; echo "dspsr -F 128:D -L 1 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
echo TC05.2/DS03/b
dspsr -F 128:D -L 8 $DS_BASE/DS03/$DS03 ; echo "dspsr -F 128:D -L 8 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
echo TC05.2/DS03/c
dspsr -F 128:D -L 30 $DS_BASE/DS03/$DS03 ; echo "dspsr -F 128:D -L 30 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code

echo TC05.2/DS04/a
dspsr -F 128:D -L 1 $DS_BASE/DS04/$DS04 ; echo "dspsr -F 128:D -L 1 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
echo TC05.2/DS04/b
dspsr -F 128:D -L 8 $DS_BASE/DS04/$DS04 ; echo "dspsr -F 128:D -L 8 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
echo TC05.2/DS04/c
dspsr -F 128:D -L 30 $DS_BASE/DS04/$DS04 ; echo "dspsr -F 128:D -L 30 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code

export TESTCASE="$TEST_EXECUTION/TC06.1"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
echo TC06.1/DS01/a
dspsr -a Timer $DS_BASE/DS01/$DS01 ; echo "dspsr -a Timer $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
echo TC06.1/DS01/b
dspsr -a PSRFITS $DS_BASE/DS01/$DS01 ; echo "dspsr -a PSRFITS $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code

echo TC06.1/DS03/a
dspsr -a Timer -U 8449 $DS_BASE/DS03/$DS03 ; echo "dspsr -a Timer -U 8449 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
echo TC06.1/DS03/b
dspsr -a PSRFITS -U 8449 $DS_BASE/DS03/$DS03 ; echo "dspsr -a PSRFITS -U 8449 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code

echo TC06.1/DS04/a
dspsr -a Timer -U 2305 $DS_BASE/DS04/$DS04 ; echo "dspsr -a Timer -U 2305 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
echo TC06.1/DS04/b
dspsr -a PSRFITS -U 2305 $DS_BASE/DS04/$DS04 ; echo "dspsr -a PSRFITS -U 2305 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code

export TESTCASE="$TEST_EXECUTION/TC06.2"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
dspsr -F 16384:D -a Timer $DS_BASE/DS01/$DS01 ; echo "dspsr -F 16384:D -a Timer $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
dspsr -F 16384:D -a PSRFITS $DS_BASE/DS01/$DS01 ; echo "dspsr -F 16384:D -a PSRFITS $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
dspsr -F 128:D -a Timer $DS_BASE/DS03/$DS03 ; echo "dspsr -F 128:D -a Timer $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
dspsr -F 128:D -a PSRFITS $DS_BASE/DS03/$DS03 ; echo "dspsr -F 128:D -a PSRFITS $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
dspsr -F 128:D -a Timer $DS_BASE/DS04/$DS04 ; echo "dspsr -F 128:D -a Timer $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
dspsr -F 128:D -a PSRFITS $DS_BASE/DS04/$DS04 ; echo "dspsr -F 128:D -a PSRFITS $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code

export TESTCASE="$TEST_EXECUTION/TC07 "
mkdir -m 0755 $TESTCASE && cd $TESTCASE
dspsr -F 128:32 $DS_BASE/DS07/$DS07; echo "dspsr -F 128:32 $DS_BASE/DS07/$DS07 exit code: $?" >> DS07.exit_code

export TESTCASE="$TEST_EXECUTION/TC08"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
dada_db -k dada -b 13600000000 -n 1; echo "dada_db -k dada -b 13600000000 -n 1 exit code: $?" >> DS01.exit_code
dada_diskdb -k dada -f $DS_BASE/DS01/$DS01; echo "dada_diskdb -k dada -f $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
echo -e "DADA INFO:\nkey dada">dada.info # input file for dspsr
dspsr dada.info; echo "dspsr dada.info exit code: $?" >> DS01.exit_code
dada_db -d -k dada # cleanup

export TESTCASE="$TEST_EXECUTION/TC09"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
digifits -nsblk 4096 -do_dedisp true -D 2.64476 -t 7.656e-06 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -do_dedisp true -D 2.64476 -t 7.656e-06 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code

export TESTCASE="$TEST_EXECUTION/TC10"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
digifits -nsblk 4096 -t 7.656e-06 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -t 7.656e-06 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code

export TESTCASE="$TEST_EXECUTION/TC11.1"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
digifits -nsblk 4096 -t 7.656e-05 -f 4 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -t 7.656e-05 -f 4 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
digifits -nsblk 4096 -t 7.656e-05 -f 64 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -t 7.656e-05 -f 64 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code

export TESTCASE="$TEST_EXECUTION/TC11.2"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
digifits -nsblk 4096 -F 16384:D -t 7.656e-05 -f 4 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -F 16384:D -t 7.656e-05 -f 4 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
digifits -nsblk 4096 -F 16384:D -t 7.656e-05 -f 64 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -F 16384:D -t 7.656e-05 -f 64 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
digifits -nsblk 4096 -F 128:D -t 64e-6 -f 4 $DS_BASE/DS03/$DS03; echo "digifits -nsblk 4096 -F 128:D -t 64e-6 -f 4 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
digifits -nsblk 4096 -F 128:D -t 64e-6 -f 64 $DS_BASE/DS03/$DS03; echo "digifits -nsblk 4096 -F 128:D -t 64e-6 -f 64 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
digifits -nsblk 4096 -F 128:D -t 64e-6 -f 4 $DS_BASE/DS04/$DS04; echo "digifits -nsblk 4096 -F 128:D -t 64e-6 -f 4 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
digifits -nsblk 4096 -F 128:D -t 64e-6 -f 64 $DS_BASE/DS04/$DS04; echo "digifits -nsblk 4096 -F 128:D -t 64e-6 -f 64 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code

export TESTCASE="$TEST_EXECUTION/TC12.1"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
digifits -nsblk 4096 -t 7.656e-05 -p 1 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -t 7.656e-05 -p 1 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
digifits -nsblk 4096 -t 7.656e-05 -p 2 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -t 7.656e-05 -p 2 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
digifits -nsblk 4096 -t 7.656e-05 -p 4 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -t 7.656e-05 -p 4 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code

export TESTCASE="$TEST_EXECUTION/TC12.2"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
digifits -nsblk 4096 -F 16384:D -t 7.656e-05 -p 1 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -F 16384:D -t 7.656e-05 -p 1 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
digifits -nsblk 4096 -F 16384:D -t 7.656e-05 -p 2 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -F 16384:D -t 7.656e-05 -p 2 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
digifits -nsblk 4096 -F 16384:D -t 7.656e-05 -p 4 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -F 16384:D -t 7.656e-05 -p 4 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
digifits -nsblk 4096 -F 128:D -t 64e-6 -p 1 $DS_BASE/DS03/$DS03; echo "digifits -nsblk 4096 -F 128:D -t 64e-6 -p 1 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
digifits -nsblk 4096 -F 128:D -t 64e-6 -p 2 $DS_BASE/DS03/$DS03; echo "digifits -nsblk 4096 -F 128:D -t 64e-6 -p 2 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
digifits -nsblk 4096 -F 128:D -t 64e-6 -p 4 $DS_BASE/DS03/$DS03; echo "digifits -nsblk 4096 -F 128:D -t 64e-6 -p 4 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
digifits -nsblk 4096 -F 128:D -t 64e-6 -p 1 $DS_BASE/DS04/$DS04; echo "digifits -nsblk 4096 -F 128:D -t 64e-6 -p 1 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
digifits -nsblk 4096 -F 128:D -t 64e-6 -p 2 $DS_BASE/DS04/$DS04; echo "digifits -nsblk 4096 -F 128:D -t 64e-6 -p 2 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
digifits -nsblk 4096 -F 128:D -t 64e-6 -p 4 $DS_BASE/DS04/$DS04; echo "digifits -nsblk 4096 -F 128:D -t 64e-6 -p 4 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code

export TESTCASE="$TEST_EXECUTION/TC13.1"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
export TSCRUNCH=2
digifits -nsblk 4096 -tscr $TSCRUNCH $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -tscr $TSCRUNCH $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
export TSCRUNCH=32
digifits -nsblk 4096 -tscr $TSCRUNCH $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -tscr $TSCRUNCH $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
export TSCRUNCH=128
digifits -nsblk 4096 -tscr $TSCRUNCH $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -tscr $TSCRUNCH $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code

export TESTCASE="$TEST_EXECUTION/TC13.2"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
export TSCRUNCH=2
digifits -nsblk 4096 -F 16384:D -tscr $TSCRUNCH $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -F 16384:D -tscr $TSCRUNCH $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
export TSCRUNCH=32
digifits -nsblk 4096 -F 16384:D -tscr $TSCRUNCH $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -F 16384:D -tscr $TSCRUNCH $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
export TSCRUNCH=128
digifits -nsblk 4096 -F 16384:D -tscr $TSCRUNCH $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -F 16384:D -tscr $TSCRUNCH $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code

export TSCRUNCH=2
digifits -nsblk 4096 -F 128:D -tscr $TSCRUNCH $DS_BASE/DS03/$DS03; echo "digifits -nsblk 4096 -F 128:D -tscr $TSCRUNCH $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
export TSCRUNCH=32
digifits -nsblk 4096 -F 128:D -tscr $TSCRUNCH $DS_BASE/DS03/$DS03; echo "digifits -nsblk 4096 -F 128:D -tscr $TSCRUNCH $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
export TSCRUNCH=128
digifits -nsblk 4096 -F 128:D -tscr $TSCRUNCH $DS_BASE/DS03/$DS03; echo "digifits -nsblk 4096 -F 128:D -tscr $TSCRUNCH $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code

export TSCRUNCH=2
digifits -nsblk 4096 -F 128:D -tscr $TSCRUNCH $DS_BASE/DS04/$DS04; echo "digifits -nsblk 4096 -F 128:D -tscr $TSCRUNCH $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
export TSCRUNCH=32
digifits -nsblk 4096 -F 128:D -tscr $TSCRUNCH $DS_BASE/DS04/$DS04; echo "digifits -nsblk 4096 -F 128:D -tscr $TSCRUNCH $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
export TSCRUNCH=128
digifits -nsblk 4096 -F 128:D -tscr $TSCRUNCH $DS_BASE/DS04/$DS04; echo "digifits -nsblk 4096 -F 128:D -tscr $TSCRUNCH $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code

export TESTCASE="$TEST_EXECUTION/TC14.1"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
digifits -nsblk 4096 -t 7.656e-05 -b 2 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -t 7.656e-05 -b 2 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
digifits -nsblk 4096 -t 7.656e-05 -b 4 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -t 7.656e-05 -b 4 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
digifits -nsblk 4096 -t 7.656e-05 -b 8 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -t 7.656e-05 -b 8 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code

export TESTCASE="$TEST_EXECUTION/TC14.2"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
digifits -nsblk 4096 -F 16384:D -t 7.656e-05 -b 2 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -F 16384:D -t 7.656e-05 -b 2 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
digifits -nsblk 4096 -F 16384:D -t 7.656e-05 -b 4 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -F 16384:D -t 7.656e-05 -b 4 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
digifits -nsblk 4096 -F 16384:D -t 7.656e-05 -b 8 $DS_BASE/DS01/$DS01; echo "digifits -nsblk 4096 -F 16384:D -t 7.656e-05 -b 8 $DS_BASE/DS01/$DS01 exit code: $?" >> DS01.exit_code
digifits -nsblk 4096 -F 128:D -t 64e-6 -b 2 $DS_BASE/DS03/$DS03; echo "digifits -nsblk 4096 -F 128:D -t 64e-6 -b 2 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
digifits -nsblk 4096 -F 128:D -t 64e-6 -b 4 $DS_BASE/DS03/$DS03; echo "digifits -nsblk 4096 -F 128:D -t 64e-6 -b 4 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
digifits -nsblk 4096 -F 128:D -t 64e-6 -b 8 $DS_BASE/DS03/$DS03; echo "digifits -nsblk 4096 -F 128:D -t 64e-6 -b 8 $DS_BASE/DS03/$DS03 exit code: $?" >> DS03.exit_code
digifits -nsblk 4096 -F 128:D -t 64e-6 -b 2 $DS_BASE/DS04/$DS04; echo "digifits -nsblk 4096 -F 128:D -t 64e-6 -b 2 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
digifits -nsblk 4096 -F 128:D -t 64e-6 -b 4 $DS_BASE/DS04/$DS04; echo "digifits -nsblk 4096 -F 128:D -t 64e-6 -b 4 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code
digifits -nsblk 4096 -F 128:D -t 64e-6 -b 8 $DS_BASE/DS04/$DS04; echo "digifits -nsblk 4096 -F 128:D -t 64e-6 -b 8 $DS_BASE/DS04/$DS04 exit code: $?" >> DS04.exit_code

export TESTCASE="$TEST_EXECUTION/TC15.1"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
dspsr -scloffs $DS_BASE/DS08/$DS08 ; echo "dspsr -scloffs $DS_BASE/DS08/$DS08 exit code: $?" >> DS08.exit_code
dspsr -scloffs $DS_BASE/DS10/$DS10 ; echo "dspsr -scloffs $DS_BASE/DS10/$DS10 exit code: $?" >> DS10.exit_code

export TESTCASE="$TEST_EXECUTION/TC15.2"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
dspsr $DS_BASE/DS08/$DS08 ; echo "dspsr $DS_BASE/DS08/$DS08 exit code: $?" >> DS08.exit_code
dspsr $DS_BASE/DS11/$DS11 ; echo "dspsr $DS_BASE/DS11/$DS11 exit code: $?" >> DS11.exit_code

export TESTCASE="$TEST_EXECUTION/TC16.1"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
dspsr -s -scloffs $DS_BASE/DS09/$DS09 ; echo "dspsr -s -scloffs $DS_BASE/DS09/$DS09 exit code: $?" >> DS09.exit_code
dspsr -s -scloffs $DS_BASE/DS11/$DS11 ; echo "dspsr -s -scloffs $DS_BASE/DS11/$DS11 exit code: $?" >> DS11.exit_code

export TESTCASE="$TEST_EXECUTION/TC16.2"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
dspsr -s $DS_BASE/DS09/$DS09 ; echo "dspsr -s $DS_BASE/DS09/$DS09 exit code: $?" >> DS09.exit_code
dspsr -s $DS_BASE/DS11/$DS11 ; echo "dspsr -s $DS_BASE/DS11/$DS11 exit code: $?" >> DS11.exit_code

export TESTCASE="$TEST_EXECUTION/TC17.1"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
dspsr -s -K -scloffs -U 851 $DS_BASE/DS09/$DS09 ; echo "dspsr -s -K -scloffs -U 851 $DS_BASE/DS09/$DS09 exit code: $?" >> DS09.exit_code
dspsr -s -K -scloffs -U 426 $DS_BASE/DS11/$DS11 ; echo "dspsr -s -K -scloffs -U 426 $DS_BASE/DS11/$DS11 exit code: $?" >> DS11.exit_code

export TESTCASE="$TEST_EXECUTION/TC17.2"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
dspsr -s -K -U 851 $DS_BASE/DS09/$DS09 ; echo "dspsr -s -K -U 851 $DS_BASE/DS09/$DS09 exit code: $?" >> DS09.exit_code
dspsr -s -K -U 426 $DS_BASE/DS11/$DS11 ; echo "dspsr -s -K -U 426 $DS_BASE/DS11/$DS11 exit code: $?" >> DS11.exit_code

export TESTCASE="$TEST_EXECUTION/TC18"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
dspsr -scloffs $DS_BASE/DS12/$DS12; echo "dspsr -scloffs $DS_BASE/DS12/$DS12 exit code: $?" >> DS12.exit_code
dspsr $DS_BASE/DS12/$DS12; echo "dspsr $DS_BASE/DS12/$DS12 exit code: $?" >> DS12.exit_code

export TESTCASE="$TEST_EXECUTION/TC19"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
dspsr -scloffs $DS_BASE/DS13/$DS13 ; echo "dspsr -scloffs $DS_BASE/DS13/$DS13 exit code: $?" >> DS13.exit_code
dspsr $DS_BASE/DS13/$DS13 ; echo "dspsr $DS_BASE/DS13/$DS13 exit code: $?" >> DS13.exit_code

export TESTCASE="$TEST_EXECUTION/TC20"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
export NBIT=2
digifits -nsblk 4096 -t 38.28e-6 -b $NBIT $DS_BASE/DS12/$DS12 -o /tmp/file.sf; echo "NBIT=$NBIT digifits -nsblk 4096 -t 38.28e-6 -b $NBIT $DS_BASE/DS12/$DS12 -o /tmp/file.sf exit code: $?" >> DS12.exit_code
dspsr -scloffs /tmp/file.sf; echo "NBIT=$NBIT dspsr -scloffs /tmp/file.sf exit code: $?" >> DS12.exit_code
export NBIT=4
digifits -nsblk 4096 -t 38.28e-6 -b $NBIT $DS_BASE/DS12/$DS12 -o /tmp/file.sf; echo "NBIT=$NBIT digifits -nsblk 4096 -t 38.28e-6 -b $NBIT $DS_BASE/DS12/$DS12 -o /tmp/file.sf exit code: $?" >> DS12.exit_code
dspsr -scloffs /tmp/file.sf; echo "NBIT=$NBIT dspsr -scloffs /tmp/file.sf exit code: $?" >> DS12.exit_code
export NBIT=8
digifits -nsblk 4096 -t 38.28e-6 -b $NBIT $DS_BASE/DS12/$DS12 -o /tmp/file.sf; echo "NBIT=$NBIT digifits -nsblk 4096 -t 38.28e-6 -b $NBIT $DS_BASE/DS12/$DS12 -o /tmp/file.sf exit code: $?" >> DS12.exit_code
dspsr -scloffs /tmp/file.sf; echo "NBIT=$NBIT dspsr -scloffs /tmp/file.sf exit code: $?" >> DS12.exit_code

export TESTCASE="$TEST_EXECUTION/TC21"
mkdir -m 0755 $TESTCASE && cd $TESTCASE
export NBIT=2
rm /tmp/file.nbit$NBIT.fil
digifil -t 4 -b $NBIT $DS_BASE/DS12/$DS12 -o /tmp/file.nbit$NBIT.fil; echo "NBIT=$NBIT digifil -t 4 -b $NBIT $DS_BASE/DS12/$DS12 -o /tmp/file.nbit$NBIT.fil exit code: $?" >> DS12.exit_code
dspsr /tmp/file.nbit$NBIT.fil; echo "NBIT=$NBIT dspsr /tmp/file.nbit$NBIT.fil exit code: $?" >> DS12.exit_code
export NBIT=4
rm /tmp/file.nbit$NBIT.fil
digifil -t 4 -b $NBIT $DS_BASE/DS12/$DS12 -o /tmp/file.nbit$NBIT.fil; echo "NBIT=$NBIT digifil -t 4 -b $NBIT $DS_BASE/DS12/$DS12 -o /tmp/file.nbit$NBIT.fil exit code: $?" >> DS12.exit_code
dspsr /tmp/file.nbit$NBIT.fil; echo "NBIT=$NBIT dspsr /tmp/file.nbit$NBIT.fil exit code: $?" >> DS12.exit_code
export NBIT=8
rm /tmp/file.nbit$NBIT.fil
digifil -t 4 -b $NBIT $DS_BASE/DS12/$DS12 -o /tmp/file.nbit$NBIT.fil; echo "NBIT=$NBIT digifil -t 4 -b $NBIT $DS_BASE/DS12/$DS12 -o /tmp/file.nbit$NBIT.fil exit code: $?" >> DS12.exit_code
dspsr /tmp/file.nbit$NBIT.fil; echo "NBIT=$NBIT dspsr /tmp/file.nbit$NBIT.fil exit code: $?" >> DS12.exit_code
export NBIT=16
rm /tmp/file.nbit$NBIT.fil
digifil -t 4 -b $NBIT $DS_BASE/DS12/$DS12 -o /tmp/file.nbit$NBIT.fil; echo "NBIT=$NBIT digifil -t 4 -b $NBIT $DS_BASE/DS12/$DS12 -o /tmp/file.nbit$NBIT.fil exit code: $?" >> DS12.exit_code
dspsr /tmp/file.nbit$NBIT.fil; echo "NBIT=$NBIT dspsr /tmp/file.nbit$NBIT.fil exit code: $?" >> DS12.exit_code
