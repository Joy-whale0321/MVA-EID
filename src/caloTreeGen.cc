#include "caloTreeGen.h"

// Tower.
#include <calobase/TowerInfo.h>
#include <calobase/TowerInfov1.h>
#include <calobase/TowerInfov2.h>
#include <calobase/TowerInfov3.h>
#include <calobase/TowerInfoContainer.h>
#include <calobase/TowerInfoContainerv1.h>
#include <calobase/TowerInfoContainerv2.h>
#include <calobase/TowerInfoContainerv3.h>
#include <calobase/TowerInfoContainerv4.h>
#include <calobase/TowerInfoDefs.h>

// Fun4All.
#include <fun4all/Fun4AllReturnCodes.h>
#include <fun4all/Fun4AllServer.h>

// Nodes.
#include <phool/PHCompositeNode.h>
#include <phool/getClass.h>
#include <phool/phool.h>
#include <phool/PHIODataNode.h>
#include <phool/PHNode.h>
#include <phool/PHNodeIterator.h>
#include <phool/PHObject.h>

#include <pdbcalbase/PdbParameterMap.h>
#include <phparameter/PHParameters.h>

// General.
#include <cstdint>
#include <iostream>
#include <map>
#include <utility>

class PHCompositeNode;
class TowerInfoContainer;
class caloTreeGen;

caloTreeGen::caloTreeGen(const std::string &name, const std::string &outfilename)
  :SubsysReco(name)
{
  verbosity = 0;
  foutname = outfilename;
  std::cout << "caloTreeGen::caloTreeGen(const std::string &name) Calling ctor" << std::endl;
}

int caloTreeGen::Init(PHCompositeNode * /*topNode*/) {
  if (verbosity > 0) std::cout << "Processing initialization: CaloEmulatorTreeMaker::Init(PHCompositeNode *topNode)" << std::endl;
  file = new TFile( foutname.c_str(), "RECREATE");
  tree = new TTree("ttree","TTree for JES calibration");

  // Tower information.
  Initialize_calo_tower();

  ievent = 0;
  return Fun4AllReturnCodes::EVENT_OK;
}

int caloTreeGen::process_event(PHCompositeNode *topNode) {
  if (verbosity >= 0) {
    if (ievent%100 == 0) std::cout << "Processing event " << ievent << std::endl;
  }

  // Tower information.
  Fill_calo_tower(topNode, "HCALIN");
  Fill_calo_tower(topNode, "HCALOUT");

  tree->Fill();
  ievent++;
  return Fun4AllReturnCodes::EVENT_OK;
}

int caloTreeGen::ResetEvent(PHCompositeNode * /*topNode*/) {
  if (Verbosity() > 1) std::cout << "Resetting the tree branches" << std::endl;

  return Fun4AllReturnCodes::EVENT_OK;
}

int caloTreeGen::End(PHCompositeNode * /*topNode*/) {
  std::cout << "caloTreeGen::End(PHCompositeNode *topNode) Saving TTree" << std::endl;
  std::cout<<"Total events: "<<ievent<<std::endl;
  file->cd();
  tree->Write();
  file->Close();
  delete file;
  std::cout << "CaloEmulatorTreeMaker complete." << std::endl;
  return Fun4AllReturnCodes::EVENT_OK;
}

////////// ********** Initialize functions ********** //////////
void caloTreeGen::Initialize_calo_tower() {
  tree->Branch("ihcal_tower_e", ihcal_tower_e, "ihcal_tower_e[24][64]/F");
  tree->Branch("ohcal_tower_e", ohcal_tower_e, "ohcal_tower_e[24][64]/F");
}

////////// ********** Fill functions ********** //////////
void caloTreeGen::Fill_calo_tower(PHCompositeNode *topNode, std::string calorimeter) 
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