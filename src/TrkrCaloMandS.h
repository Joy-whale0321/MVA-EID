// Tell emacs that this is a C++ source
//  -*- C++ -*-.
/*!
 *  \file   TrkrCaloMandS.h
 */

#ifndef TRKRCALOMANDS_H
#define TRKRCALOMANDS_H

#include <fun4all/SubsysReco.h>

#include <trackbase_historic/SvtxTrackMap.h>
#include <trackbase_historic/SvtxTrackMap_v2.h>
#include <calobase/RawClusterContainer.h>
#include <calobase/RawTowerGeomContainer.h>
#include <calobase/TowerInfoContainer.h>
#include <trackbase/TrkrHitSetContainer.h>
#include <trackbase/TrkrClusterContainer.h>

#include <g4main/PHG4Particle.h>
#include <g4main/PHG4TruthInfoContainer.h>
#include <g4main/PHG4VtxPoint.h>
#include <phhepmc/PHHepMCGenEvent.h>
#include <phhepmc/PHHepMCGenEventMap.h>

#include <TH2D.h>

#include <string>
#include <vector>

class PHCompositeNode;
class PHNode;
class TH1;
class TH2;
class TFile;
class TTree;
class PHG4TruthInfoContainer;
class PHG4Particle;
class PHHepMCGenEvent;
class PHHepMCGenEventMap;
class ActsGeometry;

class TrkrCaloMandS : public SubsysReco
{
 public:

  TrkrCaloMandS(const std::string &name = "TrkrCaloMandS", const std::string &file = "output.root");

  ~TrkrCaloMandS() override;

  /** Called during initialization.
      Typically this is where you can book histograms, and e.g.
      register them to Fun4AllServer (so they can be output to file
      using Fun4AllServer::dumpHistos() method).
   */
  int Init(PHCompositeNode *topNode) override;

  /** Called for each event.
      This is where you do the real work.
   */
  int process_event(PHCompositeNode *topNode) override;

//   int ResetEvent(PHCompositeNode *topNode) override;

  /// Called at the end of all processing.
  int End(PHCompositeNode *topNode) override;

  std::string GetTrackMapName() {return m_trackMapName;}
  void SetTrackMapName(std::string name) {m_trackMapName = name;}
  std::string GetMyTrackMapName() {return m_trackMapName_new;}
  void SetMyTrackMapName(std::string name) {m_trackMapName_new = name;}

  void writeEventDisplays( bool value ) { m_write_evt_display = value; }

  void setEventDisplayPath( std::string path ) { m_evt_display_path = path; }
  std::string getEventDisplayPath() {return m_evt_display_path;}

  void setRunDate ( std::string date ) { m_run_date = date; }
  std::string getRunDate () {return m_run_date;}

  void event_file_start(std::ofstream &jason_file_header, std::string date, int runid, int evtid);

  void doSimulation(bool set) {m_is_simulation = set;}

  void EMcalRadiusUser(bool use) {m_use_emcal_radius = use;}
  void IHcalRadiusUser(bool use) {m_use_ihcal_radius = use;}
  void OHcalRadiusUser(bool use) {m_use_ohcal_radius = use;}
  void setEMcalRadius(float r) {m_emcal_radius_user = r;}
  void setIHcalRadius(float r) {m_ihcal_radius_user = r;}
  void setOHcalRadius(float r) {m_ohcal_radius_user = r;}

  void setRawClusContEMName(std::string name) {m_RawClusCont_EM_name = name;}
  void setRawClusContTOPOName(std::string name) {m_RawClusCont_TOPO_name = name;}
  void setRawClusContHADName(std::string name) {m_RawClusCont_HAD_name = name;}
  void setRawTowerGeomContName(std::string name) {m_RawTowerGeomCont_name = name;}

  void setTrackPtLowCut(float pt) {m_track_pt_low_cut = pt;}
  void setEmcalELowCut(float e) {m_emcal_e_low_cut = e;}
  void setnMvtxClusters(int n) {m_nmvtx_low_cut = n;}
  void setnInttClusters(int n) {m_nintt_low_cut = n;}
  void setnTpcClusters(int n) {m_ntpc_low_cut = n;}
  void setnTpotClusters(int n) {m_ntpot_low_cut = n;}
  void setTrackQuality(float q) {m_track_quality = q;}
  void setdphicut(float a) {m_dphi_cut = a;};
  void setdzcut(float a) {m_dz_cut = a;};

  void Fill_Match_Info_TrkCalo(SvtxTrack* track_matched, SvtxTrackState *thisState_matched, RawCluster *EMcluster_matched);
  void Fill_calo_tower(PHCompositeNode *topNode, std::string calorimeter);

    float PiRange(float phi)
    {
        while (phi <= -M_PI) phi += 2 * M_PI;
        while (phi > M_PI)   phi -= 2 * M_PI;
        return phi;
    }

 private:
    bool checkTrack(SvtxTrack* track);

    int count_em_clusters = 0;
    int count_topo_clusters = 0;

    int m_runNumber = 0;
    int m_evtNumber = 0;
    bool m_use_emcal_radius = false;
    bool m_use_ihcal_radius = false;
    bool m_use_ohcal_radius = false;
    float m_emcal_radius_user = 93.5;
    float m_ihcal_radius_user = 117;
    float m_ohcal_radius_user = 177.423;
    bool m_is_simulation = false;
    PHG4TruthInfoContainer *m_truthInfo = nullptr;
    PHHepMCGenEventMap *m_geneventmap = nullptr;
    PHHepMCGenEvent *m_genevt = nullptr;
    SvtxTrackMap* trackMap = nullptr;
    SvtxTrackMap_v2* trackMap_new = nullptr;
    ActsGeometry* acts_Geometry = nullptr;
    RawClusterContainer* clustersEM = nullptr;
    RawClusterContainer* clustersTOPO = nullptr;
    RawClusterContainer* clustersHAD = nullptr;
    RawClusterContainer* EMCAL_RawClusters = nullptr;
    TowerInfoContainer* EMCAL_Container = nullptr;
    TowerInfoContainer* IHCAL_Container = nullptr;
    TowerInfoContainer* OHCAL_Container = nullptr;
    TrkrHitSetContainer* trkrHitSet = nullptr;
    TrkrClusterContainer* trkrContainer = nullptr;
    RawTowerGeomContainer* EMCalGeo = nullptr;
    RawTowerGeomContainer* IHCalGeo = nullptr;
    RawTowerGeomContainer* OHCalGeo = nullptr;

    std::string m_trackMapName = "SvtxTrackMap";
    std::string m_trackMapName_new = "MySvtxTrackMap";

    std::string m_RawClusCont_EM_name = "TOPOCLUSTER_EMCAL";
    std::string m_RawClusCont_TOPO_name = "TOPOCLUSTER_TOPO";
    std::string m_RawClusCont_HAD_name = "TOPOCLUSTER_HCAL";
    std::string m_RawTowerGeomCont_name = "TOWERGEOM_CEMC";
    std::string m_towerinfo_container_name = "TOWERINFO_CALIB_HCALIN";

    bool m_write_evt_display;
    std::string m_evt_display_path;
    std::string m_run_date;

    float m_track_pt_low_cut = 1;
    float m_emcal_e_low_cut = 0.5;
    float m_topo_e_low_cut = 0.1;
    int m_nmvtx_low_cut = 0;
    int m_nintt_low_cut = 0;
    int m_ntpc_low_cut = 20;
    int m_ntpot_low_cut = 0;
    float m_track_quality = 1000;
    float m_dphi_cut = 0.5;
    float m_dz_cut = 20;

    std::string _outfilename;
    TFile* file_4mva = nullptr;
    TTree* tree_4mva = nullptr;

    std::vector<float> _track_ptq;
    std::vector<float> _track_pt;
    std::vector<float> _track_px;
    std::vector<float> _track_py;
    std::vector<float> _track_pz;

    std::vector<float> _track_px_emc;
    std::vector<float> _track_py_emc;
    std::vector<float> _track_pz_emc;

    std::vector<float> _emcal_e;
    std::vector<float> _emcal_phi;
    std::vector<float> _emcal_eta;
    std::vector<float> _emcal_x;
    std::vector<float> _emcal_y;
    std::vector<float> _emcal_z;
    std::vector<float> _emcal_ecore;
    std::vector<float> _emcal_chi2;
    std::vector<float> _emcal_prob;

    std::vector<float> _ihcal_delta_eta;
    std::vector<float> _ihcal_delta_phi;

    TH2D* h2etaphibin = new TH2D("h2etaphibin", "h2etaphibin;X Axis;Y Axis;Counts", 103, -2.5, 100.5, 103, -2.5, 100.5);
    TH2D* h2tracketaphi = new TH2D("h2tracketaphi", "h2tracketaphi;X Axis;Y Axis;Counts", 400, -2, 2, 100, -7, 7);

    // Tower information.
    static const int n_hcal_tower = 1536;
    static const int n_hcal_tower_etabin = 24;
    static const int n_hcal_tower_phibin = 64;

    // HCal Tower information.
    float ihcal_tower_e[n_hcal_tower_etabin][n_hcal_tower_phibin]{};
    float ohcal_tower_e[n_hcal_tower_etabin][n_hcal_tower_phibin]{};
};  

#endif
// TRKRCALOMANDS_H
