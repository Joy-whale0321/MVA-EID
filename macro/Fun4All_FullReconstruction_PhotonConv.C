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

#include <caloreco/CaloGeomMapping.h>
#include <caloreco/CaloGeomMappingv2.h>
#include <caloreco/RawClusterBuilderTemplate.h>
#include <caloreco/RawClusterBuilderTopo.h>

#include <stdio.h>

#pragma GCC diagnostic push

#pragma GCC diagnostic ignored "-Wundefined-internal"

#include <kfparticle_sphenix/KFParticle_sPHENIX.h>
#include <kfparticle_sphenix/KshortReconstruction_local.h>

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

void Fun4All_FullReconstruction_PhotonConv(
    const int nEvents = 1,
    const std::string tpcfilename = "clusters_seeds_53744-0-0.root_dst.root",
    const std::string tpcdir = "/sphenix/user/jzhang1/TrackProduction/Reconstructed/53744/",
    const std::string calofilename = "DST_CALO_run2pp_ana462_2024p010_v001-00053744-00000.root",
    const std::string calodir = "/sphenix/lustre01/sphnxpro/production/run2pp/physics/ana462_2024p010_v001/DST_CALO/run_00053700_00053800/dst/",
    const std::string outfilename = "clusters_seeds",
    const std::string outdir = "/sphenix/user/jzhang1/testcode4all/PhotonConv/macro/root",
    const int runnumber = 53744,
    const int segment = 0,
    const int index = 0,
    const int stepsize = 10)
{
  std::string inputtpcRawHitFile = tpcdir + tpcfilename;
  std::string inputCaloFile = calodir + calofilename;

  std::pair<int, int>
      runseg = Fun4AllUtils::GetRunSegment(tpcfilename);
  //int runnumber = runseg.first;
  //int segment = runseg.second;

  string outDir = outdir + "/inReconstruction/" + to_string(runnumber) + "/";
  string makeDirectory = "mkdir -p " + outDir;
  system(makeDirectory.c_str());
  TString outfile = outDir + outfilename + "_" + runnumber + "-" + segment + "-" + index + ".root";
  std::cout<<"outfile "<<outfile<<std::endl;
  std::string theOutfile = outfile.Data();

  auto se = Fun4AllServer::instance();
  se->Verbosity(0);
  auto rc = recoConsts::instance();
  rc->set_IntFlag("RUNNUMBER", runnumber);
  rc->set_IntFlag("RUNSEGMENT", segment);

  Enable::CDB = true;
  rc->set_StringFlag("CDB_GLOBALTAG", "ProdA_2024");
  rc->set_uint64Flag("TIMESTAMP", runnumber);
  std::string geofile = CDBInterface::instance()->getUrl("Tracking_Geometry");

  Fun4AllRunNodeInputManager *ingeo = new Fun4AllRunNodeInputManager("GeoIn");
  ingeo->AddFile(geofile);
  se->registerInputManager(ingeo);

  G4MAGNET::magfield_rescale = 1;
  TrackingInit();

  auto hitsin_track = new Fun4AllDstInputManager("DSTin_track");
  hitsin_track->fileopen(inputtpcRawHitFile);
  se->registerInputManager(hitsin_track);

  auto hitsin_calo = new Fun4AllDstInputManager("DSTin_calo");
  hitsin_calo->fileopen(inputCaloFile);
  se->registerInputManager(hitsin_calo);

  Global_Reco();

  bool doEMcalRadiusCorr = true;
  auto projection = new PHActsTrackProjection("CaloProjection");
  float new_cemc_rad = 99.; // from DetailedCalorimeterGeometry, project to inner surface
  //float new_cemc_rad = 100.70;//(1-(-0.077))*93.5 recommended cemc radius at shower max
  //float new_cemc_rad = 99.1;//(1-(-0.060))*93.5
  //float new_cemc_rad = 97.6;//(1-(-0.044))*93.5, (0.041+0.047)/2=0.044
  if (doEMcalRadiusCorr)
  {
    projection->setLayerRadius(SvtxTrack::CEMC, new_cemc_rad);
  }
  se->registerSubsystem(projection);

  /////////////////////////////////////////////////////
  // Set status of CALO towers, Calibrate towers,  Cluster
  //Process_Calo_Calib();

  //my calo reco
  std::cout<<"Begin my calo reco"<<std::endl;
  // Load the modified geometry
  CaloGeomMappingv2 *cgm = new CaloGeomMappingv2();
  cgm->set_detector_name("CEMC");
  cgm->setTowerGeomNodeName("TOWERGEOM_CEMCv3");
  se->registerSubsystem(cgm);

  //////////////////
  // Clusters
  std::cout << "Building clusters" << std::endl;
  RawClusterBuilderTemplate *ClusterBuilder = new RawClusterBuilderTemplate("EmcRawClusterBuilderTemplate");
  ClusterBuilder->Detector("CEMC");
  ClusterBuilder->setUseRawTowerGeomv5(true);
  ClusterBuilder->setProjectToInnerSurface(true);
  ClusterBuilder->set_threshold_energy(0.070);  // for when using basic calibration
  std::string emc_prof = getenv("CALIBRATIONROOT");
  emc_prof += "/EmcProfile/CEMCprof_Thresh30MeV.root";
  ClusterBuilder->LoadProfile(emc_prof);
  ClusterBuilder->set_UseTowerInfo(1);  // to use towerinfo objects rather than old RawTower
  se->registerSubsystem(ClusterBuilder);

  //For particle flow studies
  RawClusterBuilderTopo* ClusterBuilder2 = new RawClusterBuilderTopo("EMcalRawClusterBuilderTopo2");
  ClusterBuilder2->Verbosity(0);
  ClusterBuilder2->set_nodename("TOPOCLUSTER_HCAL");
  ClusterBuilder2->set_enable_HCal(true);
  ClusterBuilder2->set_enable_EMCal(false);
  //ClusterBuilder2->set_noise(0.0025, 0.006, 0.03);
  ClusterBuilder2->set_noise(0.01, 0.03, 0.03);
  ClusterBuilder2->set_significance(4.0, 2.0, 1.0);
  ClusterBuilder2->allow_corner_neighbor(true);
  ClusterBuilder2->set_do_split(true);
  ClusterBuilder2->set_minE_local_max(1.0, 2.0, 0.5);
  ClusterBuilder2->set_R_shower(0.025);
  se->registerSubsystem(ClusterBuilder2);

  
  TrackCaloMatch *tcm = new TrackCaloMatch("Tracks_Calo_Match");
  tcm->SetMyTrackMapName("MySvtxTrackMap");
  tcm->writeEventDisplays(false);
  tcm->EMcalRadiusUser(doEMcalRadiusCorr);
  tcm->setEMcalRadius(new_cemc_rad);
  tcm->setdphicut(0.15);
  tcm->setdzcut(10);
  tcm->setTrackPtLowCut(0.2);
  tcm->setEmcalELowCut(0.1);
  tcm->setnTpcClusters(20);
  tcm->setTrackQuality(1000);
  tcm->setRawClusContEMName("CLUSTERINFO_CEMC");
  tcm->setRawTowerGeomContName("TOWERGEOM_CEMCv3");
  se->registerSubsystem(tcm);

  TString photonconv_kfp_likesign_outfile = theOutfile + "_photonconv_kfp_likesign.root";
  std::string photonconv_kfp_likesign_string(photonconv_kfp_likesign_outfile.Data());
  KFPReco("PhotonConvKFPReco_likesign", "[gamma -> e^+ e^+]cc", photonconv_kfp_likesign_string, "MySvtxTrackMap", "PhotonConv_likesign");

  TString photonconv_kfp_unlikesign_outfile = theOutfile + "_photonconv_kfp_unlikesign.root";
  std::string photonconv_kfp_unlikesign_string(photonconv_kfp_unlikesign_outfile.Data());
  KFPReco("PhotonConvKFPReco_unlikesign", "gamma -> e^+ e^-", photonconv_kfp_unlikesign_string, "MySvtxTrackMap", "PhotonConv_unlikesign");

  TString track2calo_outfile = theOutfile + "_track2calo.root";
  std::string track2calo_string(track2calo_outfile.Data());
  TrackToCalo *ttc = new TrackToCalo("Tracks_And_Calo", track2calo_string);
  ttc->EMcalRadiusUser(doEMcalRadiusCorr);
  ttc->setEMcalRadius(new_cemc_rad);
  ttc->setKFPtrackMapName("PhotonConv_unlikesign_SvtxTrackMap");
  ttc->setKFPContName("PhotonConv_unlikesign_KFParticle_Container");
  ttc->anaTrkrInfo(false); // general track QA
  ttc->anaCaloInfo(false); // general calo QA
  ttc->doTrkrCaloMatching(false); // SvtxTrack match with calo
  ttc->doTrkrCaloMatching_KFP(true); // KFP selected trck match with calo
  ttc->setTrackPtLowCut(0.2);
  ttc->setEmcalELowCut(0.1);
  ttc->setnTpcClusters(20);
  ttc->setTrackQuality(1000);
  ttc->setRawClusContEMName("CLUSTERINFO_CEMC");
  ttc->setRawTowerGeomContName("TOWERGEOM_CEMCv3");
  se->registerSubsystem(ttc); 

  se->skip(stepsize*index);
  se->run(nEvents);
  se->End();
  se->PrintTimer();

  ifstream file_photonconv_kfp_likesign(photonconv_kfp_likesign_string.c_str(), ios::binary | ios::ate);
  if (file_photonconv_kfp_likesign.good() && (file_photonconv_kfp_likesign.tellg() > 100))
  {
    string outputDirMove = outdir + "/Reconstructed/" + to_string(runnumber) + "/";
    string makeDirectoryMove = "mkdir -p " + outputDirMove;
    system(makeDirectoryMove.c_str());
    string moveOutput = "mv " + photonconv_kfp_likesign_string + " " + outputDirMove;
    std::cout << "moveOutput: " << moveOutput << std::endl;
    system(moveOutput.c_str());
  }

  ifstream file_photonconv_kfp_unlikesign(photonconv_kfp_unlikesign_string.c_str(), ios::binary | ios::ate);
  if (file_photonconv_kfp_unlikesign.good() && (file_photonconv_kfp_unlikesign.tellg() > 100))
  {
    string outputDirMove = outdir + "/Reconstructed/" + to_string(runnumber) + "/";
    string makeDirectoryMove = "mkdir -p " + outputDirMove;
    system(makeDirectoryMove.c_str());
    string moveOutput = "mv " + photonconv_kfp_unlikesign_string + " " + outputDirMove;
    std::cout << "moveOutput: " << moveOutput << std::endl;
    system(moveOutput.c_str());
  }

  ifstream file_track2calo(track2calo_string.c_str(), ios::binary | ios::ate);
  if (file_track2calo.good() && (file_track2calo.tellg() > 100))
  {
    string outputDirMove = outdir + "/Reconstructed/" + to_string(runnumber) + "/";
    string makeDirectoryMove = "mkdir -p " + outputDirMove;
    system(makeDirectoryMove.c_str());
    string moveOutput = "mv " + track2calo_string + " " + outputDirMove;
    std::cout << "moveOutput: " << moveOutput << std::endl;
    system(moveOutput.c_str());
  }

  delete se;
  std::cout << "Finished" << std::endl;
  gSystem->Exit(0);
}

void KFPReco(std::string module_name = "KFPReco", std::string decaydescriptor = "K_S0 -> pi^+ pi^-", std::string outfile = "KFP.root", std::string trackmapName = "SvtxTrackMap", std::string containerName = "KFParticle")
{
  auto se = Fun4AllServer::instance();
  //KFParticle setup
  KFParticle_sPHENIX *kfparticle = new KFParticle_sPHENIX(module_name);
  kfparticle->Verbosity(0);
  kfparticle->setDecayDescriptor(decaydescriptor);

  kfparticle->setTrackMapNodeName(trackmapName);
  kfparticle->setContainerName(containerName);

  //Basic node selection and configuration
  kfparticle->magFieldFile("FIELDMAP_TRACKING");
  kfparticle->getAllPVInfo(false);
  kfparticle->allowZeroMassTracks(true);
  kfparticle->getDetectorInfo(true);
  kfparticle->useFakePrimaryVertex(false);
  kfparticle->saveDST();

  kfparticle->constrainToPrimaryVertex(true);
  kfparticle->setMotherIPchi2(FLT_MAX);
  kfparticle->setFlightDistancechi2(-1.);
  kfparticle->setMinDIRA(-1.1);
  kfparticle->setDecayLengthRange(0., FLT_MAX);
  kfparticle->setDecayTimeRange(-1*FLT_MAX, FLT_MAX);

  //Track parameters
  kfparticle->setMinMVTXhits(0);
  //kfparticle->setMinTPChits(20);
  kfparticle->setMinTPChits(0);
  kfparticle->setMinimumTrackPT(0.2);
  kfparticle->setMaximumTrackPTchi2(FLT_MAX);
  kfparticle->setMinimumTrackIPchi2(-1.);
  kfparticle->setMinimumTrackIP(-1.);
  //kfparticle->setMaximumTrackchi2nDOF(100.);
  kfparticle->setMaximumTrackchi2nDOF(FLT_MAX);

  //Vertex parameters
  //kfparticle->setMaximumVertexchi2nDOF(50);
  kfparticle->setMaximumVertexchi2nDOF(FLT_MAX);
  //kfparticle->setMaximumDaughterDCA(1.);
  kfparticle->setMaximumDaughterDCA(FLT_MAX);

  //Parent parameters
  kfparticle->setMotherPT(0);
  kfparticle->setMinimumMass(-1);
  kfparticle->setMaximumMass(10);
  //kfparticle->setMaximumMotherVertexVolume(0.1);
  kfparticle->setMaximumMotherVertexVolume(FLT_MAX);

  kfparticle->setOutputName(outfile);

  se->registerSubsystem(kfparticle);
}
