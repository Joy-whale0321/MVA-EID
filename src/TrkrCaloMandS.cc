/*!
 *  \file   TrkrCaloMandS.h
 */

#include "TrkrCaloMandS.h"

#include <calobase/RawClusterContainer.h>
#include <calobase/RawTowerGeomContainer.h>
#include <calobase/RawCluster.h>
#include <calobase/RawClusterUtility.h>
#include <calobase/RawTowerDefs.h>
#include <calobase/RawTowerGeom.h>
#include <calobase/TowerInfoContainer.h>
#include <calobase/TowerInfo.h>
#include <calobase/TowerInfoDefs.h>

#include <ffarawobjects/Gl1Packet.h>
#include <ffaobjects/EventHeaderv1.h>

#include <fun4all/Fun4AllReturnCodes.h>

#include <globalvertex/GlobalVertex.h>
#include <globalvertex/GlobalVertexMap.h>
#include <globalvertex/MbdVertexMap.h>
#include <globalvertex/MbdVertex.h>
#include <globalvertex/SvtxVertexMap.h>
#include <globalvertex/SvtxVertex.h>

#include <phool/getClass.h>
#include <phool/PHCompositeNode.h>

#include <trackbase/TrkrCluster.h>
#include <trackbase/TrkrClusterContainer.h>
#include <trackbase/TrkrClusterCrossingAssocv1.h>
#include <trackbase/TrkrDefs.h>
#include <trackbase/TrkrHitSetContainer.h>
#include <trackbase/TrkrHitSet.h>
#include <trackbase_historic/SvtxTrackMap.h>
#include <trackbase_historic/SvtxTrackMap_v1.h>
#include <trackbase_historic/SvtxTrackMap_v2.h>
#include <trackbase_historic/SvtxTrackState_v1.h>
#include <trackbase_historic/TrackSeedContainer.h>
#include <trackbase_historic/TrackSeed.h>
#include <trackbase_historic/TrackAnalysisUtils.h>
#include <trackreco/ActsPropagator.h>

#include <Acts/Geometry/GeometryIdentifier.hpp>
#include <Acts/MagneticField/ConstantBField.hpp>
#include <Acts/MagneticField/MagneticFieldProvider.hpp>
#include <Acts/Surfaces/CylinderSurface.hpp>
#include <Acts/Surfaces/PerigeeSurface.hpp>
#include <Acts/Geometry/TrackingGeometry.hpp>

#include <phool/getClass.h>
#include <phool/PHCompositeNode.h>
#include <phool/PHNodeIterator.h>

#include <CLHEP/Vector/ThreeVector.h>
#include <math.h>
#include <vector>

#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TH2D.h>

#include <boost/format.hpp>
#include <boost/math/special_functions/sign.hpp>

namespace
{
    //! get cluster keys from a given track
    std::vector<TrkrDefs::cluskey> get_cluster_keys(SvtxTrack* track)
    {
        std::vector<TrkrDefs::cluskey> out;
        for (const auto& seed : {track->get_silicon_seed(), track->get_tpc_seed()})
        {
            if (seed)
            {
                std::copy(seed->begin_cluster_keys(), seed->end_cluster_keys(), std::back_inserter(out));
            }
        }
        return out;
    }

    /// return number of clusters of a given type that belong to a tracks
    template <int type>
    int count_clusters(const std::vector<TrkrDefs::cluskey>& keys)
    {
        return std::count_if(keys.begin(), keys.end(),
                            [](const TrkrDefs::cluskey& key)
                            { return TrkrDefs::getTrkrId(key) == type; });
    }
}

//____________________________________________________________________________..
TrkrCaloMandS::TrkrCaloMandS(const std::string &name, const std::string &file):
    SubsysReco(name),
    _outfilename(file),
    file_4mva(nullptr)
{
    std::cout << "TrkrCaloMandS::TrkrCaloMandS(const std::string &name) Calling ctor" << std::endl;
}

//____________________________________________________________________________..
TrkrCaloMandS::~TrkrCaloMandS()
{
    std::cout << "TrkrCaloMandS::~TrkrCaloMandS() Calling dtor" << std::endl;
}

//____________________________________________________________________________..
int TrkrCaloMandS::Init(PHCompositeNode *topNode)
{
    std::cout << topNode << std::endl;
    std::cout << "TrkrCaloMandS::Init(PHCompositeNode *topNode) Initializing" << std::endl;

    PHNodeIterator iter(topNode);

    PHCompositeNode* dstNode = dynamic_cast<PHCompositeNode*>(iter.findFirst("PHCompositeNode", "DST"));

    if (!dstNode)
    {
      std::cerr << "DST node is missing, quitting" << std::endl;
      throw std::runtime_error("Failed to find DST node in TrkrCaloMandS::Init");
    }

    PHNodeIterator dstIter(topNode);
    PHCompositeNode* svtxNode = dynamic_cast<PHCompositeNode*>(dstIter.findFirst("PHCompositeNode", "SVTX"));
    if (!svtxNode)
    {
        svtxNode = new PHCompositeNode("SVTX");
        dstNode->addNode(svtxNode);
    }

    trackMap_new = findNode::getClass<SvtxTrackMap_v2>(topNode, m_trackMapName_new);
    if(!trackMap_new)
    {
        trackMap_new = new SvtxTrackMap_v2;
        PHIODataNode<PHObject>* trackNode = new PHIODataNode<PHObject>(trackMap_new, m_trackMapName_new, "PHObject");
        svtxNode->addNode(trackNode);
        trackMap_new->clear();
    }

    // vector to store information for mva-eid
    _track_ptq.clear();
    _track_pt.clear();
    _track_px.clear();
    _track_py.clear();
    _track_pz.clear();

    _track_px_emc.clear();
    _track_py_emc.clear();
    _track_pz_emc.clear();

    _emcal_e.clear();
    _emcal_phi.clear();
    _emcal_eta.clear();
    _emcal_x.clear();
    _emcal_y.clear();
    _emcal_z.clear();
    _emcal_ecore.clear();
    _emcal_chi2.clear();
    _emcal_prob.clear();

    _ihcal_delta_eta.clear();
    _ihcal_delta_phi.clear();

    // write a tree to store data for mva-eid
    delete file_4mva;
    file_4mva = new TFile(_outfilename.c_str(), "RECREATE");

    tree_4mva = new TTree("tree_4mva", "MVA-EID pico dst info");

    tree_4mva->Branch("track_ptq", &_track_ptq);
    tree_4mva->Branch("track_pt", &_track_pt);
    tree_4mva->Branch("track_px", &_track_px);
    tree_4mva->Branch("track_py", &_track_py);
    tree_4mva->Branch("track_pz", &_track_pz);

    tree_4mva->Branch("track_px_emc", &_track_px_emc);
    tree_4mva->Branch("track_py_emc", &_track_py_emc);
    tree_4mva->Branch("track_pz_emc", &_track_pz_emc);

    tree_4mva->Branch("emcal_e", &_emcal_e);
    tree_4mva->Branch("emcal_phi", &_emcal_phi);
    tree_4mva->Branch("emcal_eta", &_emcal_eta);
    tree_4mva->Branch("emcal_x", &_emcal_x);
    tree_4mva->Branch("emcal_y", &_emcal_y);
    tree_4mva->Branch("emcal_z", &_emcal_z);
    tree_4mva->Branch("emcal_ecore", &_emcal_ecore);
    tree_4mva->Branch("emcal_chi2", &_emcal_chi2);
    tree_4mva->Branch("emcal_prob", &_emcal_prob);

    tree_4mva->Branch("ihcal_delta_eta", &_ihcal_delta_eta);
    tree_4mva->Branch("ihcal_delta_phi", &_ihcal_delta_phi);

    return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
int TrkrCaloMandS::process_event(PHCompositeNode* topNode)
{
    PHNodeIterator nodeIter(topNode);
    PHNode* evtNode = dynamic_cast<PHNode*>(nodeIter.findFirst("EventHeader"));
    
    // std::cout<<"1111111111111111111111"<<std::endl;

    if (evtNode)
    {
        EventHeaderv1* evtHeader = findNode::getClass<EventHeaderv1>(topNode, "EventHeader");
        m_runNumber = evtHeader->get_RunNumber();
        m_evtNumber = evtHeader->get_EvtSequence();
    }
    else
    {
        m_runNumber = m_evtNumber = -1;
    }
  
    std::cout << "TrkrCaloMandS::process_event run " << m_runNumber << " event " << m_evtNumber << std::endl;
  
    if(!trackMap)
    {
        trackMap = findNode::getClass<SvtxTrackMap>(topNode, m_trackMapName);
        if(!trackMap)
        {
            std::cout << "TrkrCaloMandS::process_event " << m_trackMapName << " not found! Aborting!" << std::endl;
            return Fun4AllReturnCodes::ABORTEVENT;
        }
    }
  
    if (!acts_Geometry)
    {
        acts_Geometry = findNode::getClass<ActsGeometry>(topNode, "ActsGeometry");
        if (!acts_Geometry)
        {
            std::cout << "TrkrCaloMandS::process_event ActsGeometry not found! Aborting!" << std::endl;
            return Fun4AllReturnCodes::ABORTEVENT;
        }
    }
  
    if (!clustersEM)
    {
        clustersEM = findNode::getClass<RawClusterContainer>(topNode, m_RawClusCont_EM_name);
        if (!clustersEM)
        {
            std::cout << "TrkrCaloMandS::process_event " << m_RawClusCont_EM_name << " not found! Aborting!" << std::endl;
            return Fun4AllReturnCodes::ABORTEVENT;
        }
    }

    if (!clustersTOPO)
    {
        clustersTOPO = findNode::getClass<RawClusterContainer>(topNode, m_RawClusCont_TOPO_name);
        if (!clustersTOPO)
        {
            std::cout << "TrkrCaloMandS::process_event " << m_RawClusCont_TOPO_name << " not found! Aborting!" << std::endl;
            return Fun4AllReturnCodes::ABORTEVENT;
        }
    }

    if (!IHCAL_Container)
    {
        IHCAL_Container = findNode::getClass<TowerInfoContainer>(topNode, m_towerinfo_container_name);
        if (!IHCAL_Container)
        {
            std::cout << "TrkrCaloMandS::process_event " << m_towerinfo_container_name << " not found! Aborting!" << std::endl;
            return Fun4AllReturnCodes::ABORTEVENT;
        }
    }
  
    if(!trkrContainer)
    {
        trkrContainer = findNode::getClass<TrkrClusterContainer>(topNode, "TRKR_CLUSTER");
        if(!trkrContainer)
        {
            std::cout << "TrkrCaloMandS::process_event TRKR_CLUSTER not found! Aborting!" << std::endl;
            return Fun4AllReturnCodes::ABORTEVENT;
        }
    }
  
    if(!EMCalGeo)
    {
        EMCalGeo = findNode::getClass<RawTowerGeomContainer>(topNode, m_RawTowerGeomCont_name);
        if(!EMCalGeo)
        {
            std::cout << "TrkrCaloMandS::process_event " << m_RawTowerGeomCont_name << " not found! Aborting!" << std::endl;
            return Fun4AllReturnCodes::ABORTEVENT;
        }
    }

    if(!IHCalGeo)
    {
        IHCalGeo = findNode::getClass<RawTowerGeomContainer>(topNode, "TOWERGEOM_HCALIN");
        if (!IHCalGeo)
        {
            std::cout << "TrkrCaloMandS::process_event " << "TOWERGEOM_HCALIN" << " not found! Aborting!" << std::endl;
            return Fun4AllReturnCodes::ABORTEVENT;
        }
    }
  
    if(m_is_simulation)
    {
      if(!m_truthInfo)
      {
        m_truthInfo = findNode::getClass<PHG4TruthInfoContainer>(topNode, "G4TruthInfo");
        if(!m_truthInfo)
        {
          std::cout << "TrkrCaloMandS::process_event G4TruthInfo not found! Aborting!" << std::endl;
          return Fun4AllReturnCodes::ABORTEVENT;
        }
      }
    
      if(!m_geneventmap)
      {
        m_geneventmap = findNode::getClass<PHHepMCGenEventMap>(topNode, "PHHepMCGenEventMap");
        if(!m_geneventmap)
        {
          std::cout << "TrkrCaloMandS::process_event PHHepMCGenEventMap not found! Aborting!" << std::endl;
          return Fun4AllReturnCodes::ABORTEVENT;
        }
      }
    
      if (m_truthInfo)
      {
        PHG4TruthInfoContainer::ConstRange range = m_truthInfo->GetParticleRange();
        if (Verbosity()>1) {std::cout << "m_truthInfo size = " << m_truthInfo->size() << std::endl;}
        for (PHG4TruthInfoContainer::ConstIterator iter = range.first; iter != range.second; ++iter)
        {
          PHG4Particle* g4particle = iter->second;
          int this_pid = g4particle->get_pid();
          if (this_pid == -11 || this_pid == 11)
          {
            if (Verbosity()>1) {std::cout << "found daughter particle e+/e-" << std::endl;}
            
            PHG4Particle* mother = nullptr;
            if (g4particle->get_parent_id() != 0)
            {
              mother = m_truthInfo->GetParticle(g4particle->get_parent_id());
              if (abs(mother->get_pid())==22)
              {
                float mother_e = mother->get_e();
                float mother_pt = sqrt( (mother->get_px())*(mother->get_px()) + (mother->get_py())*(mother->get_py()));
                float mother_eta = asinh(mother->get_pz()/sqrt(mother->get_px()*mother->get_px() + mother->get_py()*mother->get_py()));
                if (Verbosity()>1) {std::cout << "daughter pid = " << this_pid << " track id = " << g4particle->get_track_id() << " mother is gamma track id= " <<    mother->get_track_id() << " E = " << mother_e << " pT = " << mother_pt << " eta = " << mother_eta << std::endl;}
              }
            }
          }
        }
      }    
    
    }
  
    double caloRadiusEMCal;
    if (m_use_emcal_radius)
    {
        caloRadiusEMCal = m_emcal_radius_user;
    }
    else
    {
        caloRadiusEMCal = EMCalGeo->get_radius();
    }

    double caloRadiusIHCal;
    if (m_use_ihcal_radius)
    {
        caloRadiusIHCal = m_ihcal_radius_user;
    }
    else
    {
        caloRadiusIHCal = 117;
    }

    // get the hcal 2d energy map
    Fill_calo_tower(topNode, "HCALIN");
  
    SvtxTrackState *cemcState = nullptr;
    SvtxTrackState *ihcalState = nullptr;
    SvtxTrack *track = nullptr;
    TrackSeed *tpc_seed = nullptr;
    TrkrCluster *trkrCluster = nullptr;

    _track_ptq.clear();
    _track_pt.clear();
    _track_px.clear();
    _track_py.clear();
    _track_pz.clear();

    _track_px_emc.clear();
    _track_py_emc.clear();
    _track_pz_emc.clear();

    _emcal_e.clear();
    _emcal_phi.clear();
    _emcal_eta.clear();
    _emcal_x.clear();
    _emcal_y.clear();
    _emcal_z.clear();
    _emcal_ecore.clear();
    _emcal_chi2.clear();
    _emcal_prob.clear();

    _ihcal_delta_eta.clear();
    _ihcal_delta_phi.clear();

    int num_matched_pair = 0;
    int num_cemcstate = 0;
    int num_ihcalstate = 0;
    for (auto &iter : *trackMap)
    {
        track = iter.second;
    
        if(!checkTrack(track))
        {
          continue;
        }
      
        cemcState = track->get_state(caloRadiusEMCal);
        float _track_phi_emc = NAN;
        float _track_eta_emc = NAN;
        float _track_x_emc = NAN;
        float _track_y_emc = NAN;
        float _track_z_emc = NAN;
    
        ihcalState = track->get_state(caloRadiusIHCal);
        float _track_phi_ihc = NAN;
        float _track_eta_ihc = NAN;
        float _track_x_ihc = NAN;
        float _track_y_ihc = NAN;
        float _track_z_ihc = NAN;
        
        if(!cemcState)
        {
          continue;
        }
        else
        {
            _track_phi_emc = atan2(cemcState->get_y(), cemcState->get_x());
            _track_eta_emc = asinh(cemcState->get_z()/sqrt(cemcState->get_x()*cemcState->get_x() + cemcState->get_y()*cemcState->get_y()));
            _track_x_emc = cemcState->get_x();
            _track_y_emc = cemcState->get_y();
            _track_z_emc = cemcState->get_z();

            num_cemcstate += 1;
        }
    
        if(!ihcalState)
        {
            continue;
        }
        else
        {
            _track_phi_ihc = atan2(ihcalState->get_y(), ihcalState->get_x());
            _track_eta_ihc = asinh(ihcalState->get_z()/sqrt(ihcalState->get_x()*ihcalState->get_x() + ihcalState->get_y()*ihcalState->get_y()));
            _track_x_ihc = ihcalState->get_x();
            _track_y_ihc = ihcalState->get_y();
            _track_z_ihc = ihcalState->get_z();

            num_ihcalstate += 1;

            // // from track2ihcal
            int etabin = IHCalGeo->get_etabin(_track_eta_ihc);
            int phibin = IHCalGeo->get_phibin(_track_phi_ihc);

            float eta_center = IHCalGeo->get_etacenter(etabin);
            float phi_center = IHCalGeo->get_phicenter(phibin);
            
            // _ihcal_delta_eta.push_back((eta_center - _track_eta_ihc));
            // _ihcal_delta_phi.push_back(PiRange(phi_center - _track_phi_ihc));

            // std::cout << "etabin and phibin are: " << etabin << ", " << phibin <<std::endl;

            // if (fabs(eta_center - _track_eta_ihc) > 0.05)
            // {
            //     std::cout << "eta=" << _track_eta_ihc << ", phi=" << _track_phi_ihc 
            //               << " corresponds to iHCal etabin=" << etabin 
            //               << ", phibin=" << phibin 
            //               << " . ihcal eta phi"<< eta_center << ", " << phi_center << std::endl;

            //     h2tracketaphi->Fill(_track_eta_ihc,_track_phi_ihc);
            //     h2etaphibin->Fill(etabin,phibin);
            // }
        }

        bool is_match = false; // ****************************
        
        RawCluster *cluster = nullptr;
        
        RawClusterContainer::Range begin_end_EMC = clustersEM->getClusters();
        RawClusterContainer::Iterator clusIter_EMC;
    
        int match_emc_cluster = 0;
        /// Loop over the EMCal clusters
        for (clusIter_EMC = begin_end_EMC.first; clusIter_EMC != begin_end_EMC.second; ++clusIter_EMC)
        {
            cluster = clusIter_EMC->second;
            if(cluster->get_energy() < m_emcal_e_low_cut) // default 0.5 GeV
            {
                continue;
            }
          
            float _emcal_phi_tem = atan2(cluster->get_y(), cluster->get_x());
            float _emcal_eta_tem = asinh(cluster->get_z()/sqrt(cluster->get_x()*cluster->get_x() + cluster->get_y()*cluster->get_y()));
            float _emcal_x_tem = cluster->get_x();
            float _emcal_y_tem = cluster->get_y();
            float radius_scale = caloRadiusEMCal / sqrt(_emcal_x_tem*_emcal_x_tem+_emcal_y_tem*_emcal_y_tem);
            float _emcal_z_tem = radius_scale*cluster->get_z();
            
            float dphi = PiRange(_track_phi_emc - _emcal_phi_tem);
            float dz = _track_z_emc - _emcal_z_tem;
          
            if(fabs(dphi)<m_dphi_cut && fabs(dz)<m_dz_cut) // default: m_dphi_cut = 0.5, m_dz_cut = 20;
            {
                match_emc_cluster += 1;
                // if(match_emc_cluster>1.1) std::cout << "match cluster > 1. "<< std::endl;

                std::cout<<"EM temple cluster phi and eta: "<< _emcal_phi_tem << ", "<< _emcal_z_tem <<std::endl;
                count_em_clusters += 1;

                is_match = true;
	            if (Verbosity() > 2)
	            {
                    std::cout<<"matched tracks!!!"<<std::endl;
                    std::cout<<"emcal x = "<<_emcal_x_tem<<" , y = "<<_emcal_y_tem<<" , z = "<<_emcal_z_tem<<" , phi = "<<_emcal_phi_tem<<" , eta = "<<_emcal_eta_tem<<std::endl;
                    std::cout<<"track projected x = "<<_track_x_emc<<" , y = "<<_track_y_emc<<" , z = "<<_track_z_emc<<" , phi = "<<_track_phi_emc<<" , eta = "<<_track_eta_emc<<std::endl;
                    std::cout<<"track px = "<<track->get_px()<<" , py = "<<track->get_py()<<" , pz = "<<track->get_pz()<<" , pt = "<<track->get_pt()<<" , p = "<<track->get_p()<<" , charge = "<<track->get_charge()<<std::endl;
                }
                Fill_Match_Info_TrkCalo(track, cemcState, cluster);       
            }
        }

        // Loop over the HCal(Topo) clusters ------------------------------------
        RawCluster *cluster_topo = nullptr;
        RawClusterContainer::Range begin_end_TOPO = clustersTOPO->getClusters();
        RawClusterContainer::Iterator clusIter_TOPO;
        for (clusIter_TOPO = begin_end_TOPO.first; clusIter_TOPO != begin_end_TOPO.second; ++clusIter_TOPO)
        {
            cluster_topo = clusIter_TOPO->second;
            if(cluster_topo->get_energy() < m_topo_e_low_cut) // default 0.5 GeV
            {
                continue;
            }

            double caloRadiusTopo = caloRadiusIHCal;
          
            float _topo_phi_tem = atan2(cluster_topo->get_y(), cluster_topo->get_x());
            float _topo_eta_tem = asinh(cluster_topo->get_z()/sqrt(cluster_topo->get_x()*cluster_topo->get_x() + cluster_topo->get_y()*cluster_topo->get_y()));
            float _topo_x_tem = cluster_topo->get_x();
            float _topo_y_tem = cluster_topo->get_y();
            double _topo_R = sqrt(_topo_x_tem*_topo_x_tem + _topo_y_tem*_topo_y_tem);

            float radius_scale = caloRadiusTopo / _topo_R;
            float _topo_z_tem = radius_scale*cluster_topo->get_z();
            
            bool em_on_topo = false;
            bool ih_on_topo = false;
            bool oh_on_topo = false;
            RawCluster::TowerConstRange towers = cluster_topo->get_towers();
            for (RawCluster::TowerConstIterator it = towers.first; it != towers.second; ++it)
            {
                unsigned int towerid = it->first;
                float fraction = it->second;
                
                RawTowerDefs::CalorimeterId calo_id = RawTowerDefs::decode_caloid(towerid);
                int ieta = RawTowerDefs::decode_index1(towerid);
                int iphi = RawTowerDefs::decode_index2(towerid);
            
                if (calo_id == RawTowerDefs::CEMC)
                {
                    em_on_topo = true;
                    // std::cout << "EMCal tower: eta = " << ieta << ", phi = " << iphi << ", fraction = " << fraction << "\n";
                }
                else if (calo_id == RawTowerDefs::HCALIN)
                {
                    ih_on_topo = true;
                    // std::cout << "IHCal tower: eta = " << ieta << ", phi = " << iphi << ", fraction = " << fraction << "\n";
                }
                else if (calo_id == RawTowerDefs::HCALOUT)
                {
                    oh_on_topo = true;
                    // std::cout << "OHCal tower: eta = " << ieta << ", phi = " << iphi << ", fraction = " << fraction << "\n";
                }
            }

            if(em_on_topo) 
            {
                float dphi = PiRange(_track_phi_emc - _topo_phi_tem);
                float dz = _track_z_emc - _topo_z_tem;
                if(fabs(dphi)<m_dphi_cut && fabs(dz)<m_dz_cut) 
                {
                    std::cout<<"EM topo cluster phi and eta: "<< _topo_phi_tem << ", "<< _topo_z_tem <<std::endl;
                    count_topo_clusters += 1;
                }
            }

            if (!oh_on_topo) continue;

            std::cout << "TOPO cluster R is: " << _topo_R << std::endl;
            
            int match_topo_cluster = 0;
            float dphi = PiRange(_track_phi_ihc - _topo_phi_tem);
            float dz = _track_z_ihc - _topo_z_tem;
            if(fabs(dphi)<m_dphi_cut && fabs(dz)<m_dz_cut) // default: m_dphi_cut = 0.5, m_dz_cut = 20;
            {
                match_topo_cluster += 1;
                if(match_topo_cluster>1.1) std::cout << "match topo cluster > 1. "<< std::endl;

                std::cout<<"corresponding topo cluster: "<<std::endl;
                std::cout<<"topo x = "<<_topo_x_tem<<" , y = "<<_topo_y_tem<<" , z = "<<_topo_z_tem<<" , phi = "<<_topo_phi_tem<<" , eta = "<<_topo_eta_tem<<std::endl;
                std::cout<<"track projected x = "<<_track_x_ihc<<" , y = "<<_track_y_ihc<<" , z = "<<_track_z_ihc<<" , phi = "<<_track_phi_ihc<<" , eta = "<<_track_eta_ihc<<std::endl;       
            }
 
            
        }

        // 可以match 的 track存个svtxmap
        if(is_match)
        {                                             
            //trackMap_new->insert(iter.second);
            trackMap_new->insertWithKey(iter.second,iter.first);
            if (Verbosity() > 1) {std::cout<<"insertWithKey iter.first = "<<iter.first<<" , track->get_id() = "<<track->get_id()<<std::endl;}
            num_matched_pair++;
        }
    }

    // std::cout<<"num_cemc_ihcal is: "<< num_cemcstate <<", "<< num_ihcalstate <<std::endl;
    
    tree_4mva->Fill();
    
    // std::cout<<"33333333333333333333333"<<std::endl;

    return Fun4AllReturnCodes::EVENT_OK;
}

//____________________________________________________________________________..
bool TrkrCaloMandS::checkTrack(SvtxTrack* track)
{
    if(!track)
    {
        return false;  
    }

    if(track->get_pt() < m_track_pt_low_cut)
    {
        return false;
    }

    if(track->get_quality() > m_track_quality)
    {
        return false;
    }

    const auto cluster_keys(get_cluster_keys(track));
    if (count_clusters<TrkrDefs::mvtxId>(cluster_keys) < m_nmvtx_low_cut)
    {
        return false;
    }
    if (count_clusters<TrkrDefs::inttId>(cluster_keys) < m_nintt_low_cut)
    {
        return false;
    }
    if (count_clusters<TrkrDefs::tpcId>(cluster_keys) < m_ntpc_low_cut)
    {
        return false;
    }
    if (count_clusters<TrkrDefs::micromegasId>(cluster_keys) < m_ntpot_low_cut)
    {
        return false;
    }

    return true;
}

//____________________________________________________________________________..
void TrkrCaloMandS::event_file_start(std::ofstream &jason_file_header, std::string date, int runid, int evtid)
{
    jason_file_header << "{\n    \"EVENT\": {\n        \"runid\": " << runid << ", \n        \"evtid\": " << evtid << ", \n        \"time\": 0, \n        \"type\": \"Collision\", \n        \"s_nn\": 0, \n        \"B\": 3.0,\n        \"pv\": [0,0,0],\n        \"runstats\": [\"sPHENIX Internal\",        \n        \"200 GeV pp\",        \n        \"" << date << ", Run " << runid << "\",        \n        \"Event #" << evtid << "\"]  \n    },\n" << std::endl;

    jason_file_header << "    \"META\": {\n       \"HITS\": {\n          \"INNERTRACKER\": {\n              \"type\": \"3D\",\n              \"options\": {\n              \"size\": 6.0,\n              \"color\": 16711680\n              } \n          },\n" << std::endl;
    jason_file_header << "          \"TRACKHITS\": {\n              \"type\": \"3D\",\n              \"options\": {\n              \"size\": 2.0,\n              \"transparent\": 0.6,\n              \"color\": 16777215\n              } \n          },\n" << std::endl;
    jason_file_header << "          \"CEMC\": {\n              \"type\": \"PROJECTIVE\",\n              \"options\": {\n                  \"rmin\": 90,\n                  \"rmax\": 136.1,\n                  \"deta\": 0.025,\n                  \"dphi\": 0.025,\n                  \"color\": 16766464,\n                  \"transparent\": 0.6,\n                  \"scaleminmax\": true\n              }\n          },\n" << std::endl;
    jason_file_header << "    \"JETS\": {\n        \"type\": \"JET\",\n        \"options\": {\n            \"rmin\": 0,\n            \"rmax\": 78,\n            \"emin\": 0,\n            \"emax\": 30,\n            \"color\": 16777215,\n            \"transparent\": 0.5 \n        }\n    }\n        }\n    }\n," << std::endl;
}

//____________________________________________________________________________..
int TrkrCaloMandS::End(PHCompositeNode *topNode)
{
    std::cout << "count clus num is: "<< count_em_clusters << ", " << count_topo_clusters << std::endl;

    file_4mva -> cd();
    tree_4mva -> Write();
    h2etaphibin->Write();
    h2tracketaphi->Write();
    file_4mva -> Close();

    std::cout << topNode << std::endl;
    std::cout << "TrkrCaloMandS::End(PHCompositeNode *topNode) Endding" << std::endl;
    return Fun4AllReturnCodes::EVENT_OK;
}


void TrkrCaloMandS::Fill_Match_Info_TrkCalo(SvtxTrack* track_matched, SvtxTrackState *cemcState_matched, RawCluster *EMcluster_matched)
{
    _track_ptq.push_back(track_matched->get_charge() * track_matched->get_pt());
    _track_pt.push_back(track_matched->get_pt());
    _track_px.push_back(track_matched->get_px());
    _track_py.push_back(track_matched->get_py());
    _track_pz.push_back(track_matched->get_pz());

    _track_px_emc.push_back(cemcState_matched->get_px());
    _track_py_emc.push_back(cemcState_matched->get_py());
    _track_pz_emc.push_back(cemcState_matched->get_pz());

    _emcal_e.push_back(EMcluster_matched->get_energy());
    // _emcal_phi.push_back(RawClusterUtility::GetAzimuthAngle(*EMcluster_matched, vertex));
    // _emcal_eta.push_back(RawClusterUtility::GetPseudorapidity(*EMcluster_matched, vertex));
    _emcal_phi.push_back(atan2(EMcluster_matched->get_y(), EMcluster_matched->get_x()));
    _emcal_eta.push_back(asinh(EMcluster_matched->get_z()/sqrt(EMcluster_matched->get_x()*EMcluster_matched->get_x() + EMcluster_matched->get_y()*EMcluster_matched->get_y())));
    
    _emcal_x.push_back(EMcluster_matched->get_x());
    _emcal_y.push_back(EMcluster_matched->get_y());
    _emcal_z.push_back(EMcluster_matched->get_z());
    _emcal_ecore.push_back(EMcluster_matched->get_ecore());
    _emcal_chi2.push_back(EMcluster_matched->get_chi2());
    _emcal_prob.push_back(EMcluster_matched->get_prob());
}


void TrkrCaloMandS::Fill_calo_tower(PHCompositeNode *topNode, std::string calorimeter) 
{
    std::string tower_info_container_name = "TOWERINFO_CALIB_" + calorimeter;
    TowerInfoContainer *_towers_calo = findNode::getClass<TowerInfoContainer>(topNode, tower_info_container_name);
    
    if (_towers_calo) 
    {
        for (int channel = 0; channel < (int)_towers_calo->size(); ++channel) 
        {
        	TowerInfo *_tower = _towers_calo->get_tower_at_channel(channel);
        	
            unsigned int towerkey = _towers_calo->encode_key(channel);
        	int etabin = _towers_calo->getTowerEtaBin(towerkey);
        	int phibin = _towers_calo->getTowerPhiBin(towerkey);
            
            float towE = _tower->get_energy();
            if (calorimeter == "HCALIN") 
            {
                ihcal_tower_e[etabin][phibin] = towE;
                // std::cout<<"towE is: "<<towE<<std::endl;  
            } 
            else if (calorimeter == "HCALOUT") 
            {
                ohcal_tower_e[etabin][phibin] = towE;
            }
        }
    } 
    else 
    { 
        std::cout << "TowerInfoContainer for " << calorimeter << " is missing" << std::endl;
    }
}


