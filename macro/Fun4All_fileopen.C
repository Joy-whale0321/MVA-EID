/*
 * This macro shows a minimum working example of running the tracking
 * hit unpackers with some basic seeding algorithms to try to put together
 * tracks. There are some analysis modules run at the end which package
 * hits, clusters, and clusters on tracks into trees for analysis.
 */

#include <fun4all/Fun4AllUtils.h>
#include <G4_ActsGeom.C>
#include <G4_Global.C>
#include <G4_Magnet.C>
#include <G4_Mbd.C>
#include <GlobalVariables.C>
#include <QA.C>
#include <Calo_Calib.C>
#include <Trkr_Clustering.C>
#include <Trkr_LaserClustering.C>
#include <Trkr_Reco.C>
#include <Trkr_RecoInit.C>
#include <Trkr_TpcReadoutInit.C>

#include <ffamodules/CDBInterface.h>

#include <fun4all/Fun4AllDstInputManager.h>
#include <fun4all/Fun4AllDstOutputManager.h>
#include <fun4all/Fun4AllInputManager.h>
#include <fun4all/Fun4AllOutputManager.h>
#include <fun4all/Fun4AllRunNodeInputManager.h>
#include <fun4all/Fun4AllServer.h>

#include <phool/recoConsts.h>

#include <cdbobjects/CDBTTree.h>

#include <tpccalib/PHTpcResiduals.h>

#include <trackingqa/InttClusterQA.h>
#include <trackingqa/MicromegasClusterQA.h>
#include <trackingqa/MvtxClusterQA.h>
#include <trackingqa/TpcClusterQA.h>
#include <tpcqa/TpcRawHitQA.h>
#include <trackingqa/TpcSeedsQA.h>

#include <trackreco/PHActsTrackProjection.h>

#include <trackingdiagnostics/TrackResiduals.h>
#include <trackingdiagnostics/TrkrNtuplizer.h>
//#include <trackingdiagnostics/KshortReconstruction.h>

#include <track_to_calo/TrackCaloMatch.h>
#include <track_to_calo/TrackToCalo.h>
#include <track_to_calo/CaloOnly.h>
#include <track_to_calo/TrackOnly.h>
#include <track_to_calo/TrkrCaloMandS.h>

#include <caloreco/CaloGeomMapping.h>
#include <caloreco/CaloGeomMappingv2.h>
#include <caloreco/RawClusterBuilderTemplate.h>
#include <caloreco/RawClusterBuilderTopo.h>

#include <stdio.h>

#pragma GCC diagnostic push

#pragma GCC diagnostic ignored "-Wundefined-internal"

#include <kfparticle_sphenix/KFParticle_sPHENIX.h>
// #include <kfparticle_sphenix/KshortReconstruction_local.h>

#pragma GCC diagnostic pop

void KFPReco(std::string module_name = "KFPReco", std::string decaydescriptor = "K_S0 -> pi^+ pi^-", std::string outfile = "KFP.root", std::string trackmapName = "SvtxTrackMap", std::string containerName = "KFParticle");

R__LOAD_LIBRARY(libkfparticle_sphenix.so)
R__LOAD_LIBRARY(libfun4all.so)
R__LOAD_LIBRARY(libffamodules.so)
R__LOAD_LIBRARY(libphool.so)
R__LOAD_LIBRARY(libcdbobjects.so)
R__LOAD_LIBRARY(libmvtx.so)
R__LOAD_LIBRARY(libintt.so)
R__LOAD_LIBRARY(libtpc.so)
R__LOAD_LIBRARY(libmicromegas.so)
R__LOAD_LIBRARY(libTrackingDiagnostics.so)
R__LOAD_LIBRARY(libtrackingqa.so)
R__LOAD_LIBRARY(libtpcqa.so)
R__LOAD_LIBRARY(libtrack_to_calo.so)
R__LOAD_LIBRARY(libtrack_reco.so)
R__LOAD_LIBRARY(libcalo_reco.so)
//R__LOAD_LIBRARY(libcalogeomtest.so)
R__LOAD_LIBRARY(libcalotrigger.so)
R__LOAD_LIBRARY(libcentrality.so)
R__LOAD_LIBRARY(libmbd.so)
R__LOAD_LIBRARY(libepd.so)
R__LOAD_LIBRARY(libzdcinfo.so)

void Fun4All_fileopen(
    const int nEvents = 5,
    const std::string tpcfilename = "clusters_seeds_53744-0-0.root_dst.root",
    const std::string tpcdir = "/sphenix/user/jzhang1/TrackProduction/Reconstructed/",
    // const std::string calofilename = "DST_CALO_run2pp_ana462_2024p012_v001-00050905-00001.root",
    // const std::string calodir = "/sphenix/lustre01/sphnxpro/production/run2pp/physics/ana462_2024p012_v001/DST_CALO/run_00050900_00051000/dst/",
    const std::string calofilename = "DST_CALO_run2pp_ana462_2024p010_v001-00053744-00001.root",
    const std::string calodir = "/sphenix/lustre01/sphnxpro/production/run2pp/physics/ana462_2024p010_v001/DST_CALO/run_00053700_00053800/dst/",
    const std::string outfilename = "clusters_seeds",
    const std::string outdir = "./root",
    const int runnumber = 53744,
    const int segment = 0,
    const int index = 0,
    const int stepsize = 10)
{
    std::string trackpro_tstem = tpcdir + std::to_string(runnumber) + "/" + tpcfilename;
    // std::string inputtpcRawHitFile = tpcdir + tpcfilename;
    std::string inputtpcRawHitFile = trackpro_tstem;
    std::string inputCaloFile = calodir + calofilename;

    std::pair<int, int> runseg = Fun4AllUtils::GetRunSegment(tpcfilename);
    //int runnumber = runseg.first;
    //int segment = runseg.second;

    auto se = Fun4AllServer::instance();
    se->Verbosity(2);
    auto rc = recoConsts::instance();
    rc->set_IntFlag("RUNNUMBER", runnumber);
    rc->set_IntFlag("RUNSEGMENT", segment);
    std::cout << ">>> RUNNUMBER is: "<< runnumber << std::endl;

    Enable::CDB = true;
    rc->set_StringFlag("CDB_GLOBALTAG", "ProdA_2024"); 
    rc->set_uint64Flag("TIMESTAMP", runnumber);
    std::string geofile = CDBInterface::instance()->getUrl("Tracking_Geometry");

    auto hitsin_calo = new Fun4AllDstInputManager("DSTin_calo");
    std::cout << ">>> After new registration" << std::endl;
    std::cout << ">>> the file need to be open is: "<< inputCaloFile << std::endl;
    hitsin_calo->fileopen(inputCaloFile);  // /sphenix/lustre01/sphnxpro/production/run2pp/physics/ana462_2024p010_v001/DST_CALO/run_00053700_00053800/dst/DST_CALO_run2pp_ana462_2024p010_v001-00053744-00000.root
    std::cout << ">>> After fileopen registration" << std::endl;
    se->registerInputManager(hitsin_calo);
    std::cout << ">>> After hitsin_calo registration" << std::endl;

    se->skip(stepsize*index);
    se->run(nEvents);
    se->End();
    se->PrintTimer();

    delete se;
    std::cout << "All Finished" << std::endl;
    gSystem->Exit(0);
}

