// Tell emacs that this is a C++ source
//  -*- C++ -*-.
#ifndef CALOTREEGEN_H
#define CALOTREEGEN_H

#include <string>
#include <vector>
#include <limits>

#include <TFile.h>
#include <TTree.h> 

#include <fun4all/SubsysReco.h>
#include <calobase/TowerInfoContainer.h>
#include <calobase/TowerInfoContainerv1.h>
#include <calobase/TowerInfoContainerv2.h>
#include <calobase/TowerInfoContainerv3.h>
#include <calobase/TowerInfoContainerv4.h>

class PHCompositeNode;

class caloTreeGen : public SubsysReco
{
 public:
  caloTreeGen(const std::string &name = "caloTreeGen", const std::string &outfilename = "output.root");
  ~caloTreeGen() override = default;
  int Init(PHCompositeNode *topNode) override;
  int process_event(PHCompositeNode *topNode) override;
  int ResetEvent(PHCompositeNode *topNode) override;
  int End(PHCompositeNode *topNode) override;

  // ********** Setters ********** //
  void SetVerbosity(int verbo) {verbosity = verbo;}

  // ********** Functions ********** //
  void Initialize_calo_tower();
  void Fill_calo_tower(PHCompositeNode *topNode, std::string calorimeter);

 private:
  // ********** General variables ********** //
  TFile *file{nullptr};
  TTree *tree{nullptr};
  std::string foutname{"output.root"};
  int verbosity{0};
  int ievent{0};

  // ********** Constants ********** //
  // Tower information.
  static const int n_hcal_tower = 1536;
  static const int n_hcal_tower_etabin = 24;
  static const int n_hcal_tower_phibin = 64;

  // ********** Tree variables ********** //
  // Tower information.
  float ihcal_tower_e[n_hcal_tower_etabin][n_hcal_tower_phibin]{};
  float ohcal_tower_e[n_hcal_tower_etabin][n_hcal_tower_phibin]{};
};

#endif