/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This macro provides a simple example on how to use the trained classifiers
/// within an analysis module
/// - Project   : TMVA - a Root-integrated toolkit for multivariate data analysis
/// - Package   : TMVA
/// - Exectuable: TMVAClassificationApplication
///
/// \macro_output
/// \macro_code
/// \author Andreas Hoecker

#include <cstdlib>
#include <vector>
#include <iostream>
#include <map>
#include <string>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TStopwatch.h"

#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodCuts.h"

#include "TApplication.h"
#include "TDirectory.h"
#include "TMath.h"
#include "TGraph.h"
#include "TGraph2D.h"
#include "TRandom.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TF3.h"
#include "TError.h"
#include "Fit/LogLikelihoodFCN.h"
#include "Fit/BasicFCN.h"
#include "Fit/BinData.h"
#include "Fit/UnBinData.h"
#include "Fit/FcnAdapter.h"
#include "Fit/FitConfig.h"
#include "Fit/FitResult.h"
#include "Fit/Fitter.h"
#include "Fit/Chi2FCN.h"
#include "Fit/PoissonLikelihoodFCN.h"
#include "TVirtualFitter.h"
#include "Math/Minimizer.h"
#include "Math/MinimizerOptions.h"
#include "Math/FitMethodFunction.h"
#include "Math/WrappedMultiTF1.h"
#include "Math/Error.h"
#include "Math/VirtualIntegrator.h"
#include "Math/GSLIntegrator.h"
#include "HFitInterface.h"
#include "Fit/FitExecutionPolicy.h"
#include "TF2.h"
#include "TF1.h"
#include "TGraphErrors.h" 
#include "TGraph.h" 
#include "TGaxis.h"
#include "TLegend.h"
#include "TText.h"
#include "TLatex.h"
#include "TAxis.h"
#include "TNtuple.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TFrame.h"
#include "TVector3.h"
#include "TFormula.h"

#include "sPhenixStyle.h"
#include "sPhenixStyle.C"

using namespace TMVA;

void TMVAClassificationApplication_eID_N( TString myMethodList = "" )
{

   //---------------------------------------------------------------
   // This loads the library
   TMVA::Tools::Instance();

   // Default MVA methods to be trained + tested
   std::map<std::string,int> Use;

   // Cut optimisation
   Use["Cuts"]            = 0;
   Use["CutsD"]           = 0;
   Use["CutsPCA"]         = 0;
   Use["CutsGA"]          = 0;
   Use["CutsSA"]          = 0;
   //
   // 1-dimensional likelihood ("naive Bayes estimator")
   Use["Likelihood"]      = 0;
   Use["LikelihoodD"]     = 0; // the "D" extension indicates decorrelated input variables (see option strings)
   Use["LikelihoodPCA"]   = 0; // the "PCA" extension indicates PCA-transformed input variables (see option strings)
   Use["LikelihoodKDE"]   = 0;
   Use["LikelihoodMIX"]   = 0;
   //
   // Mutidimensional likelihood and Nearest-Neighbour methods
   Use["PDERS"]           = 0;
   Use["PDERSD"]          = 0;
   Use["PDERSPCA"]        = 0;
   Use["PDEFoam"]         = 0;
   Use["PDEFoamBoost"]    = 0; // uses generalised MVA method boosting
   Use["KNN"]             = 0; // k-nearest neighbour method
   //
   // Linear Discriminant Analysis
   Use["LD"]              = 1; // Linear Discriminant identical to Fisher
   Use["Fisher"]          = 0;
   Use["FisherG"]         = 0;
   Use["BoostedFisher"]   = 0; // uses generalised MVA method boosting
   Use["HMatrix"]         = 0;
   //
   // Function Discriminant analysis
   Use["FDA_GA"]          = 0; // minimisation of user-defined function using Genetics Algorithm
   Use["FDA_SA"]          = 0;
   Use["FDA_MC"]          = 0;
   Use["FDA_MT"]          = 0;
   Use["FDA_GAMT"]        = 0;
   Use["FDA_MCMT"]        = 0;
   //
   // Neural Networks (all are feed-forward Multilayer Perceptrons)
   Use["MLP"]             = 0; // Recommended ANN
   Use["MLPBFGS"]         = 0; // Recommended ANN with optional training method
   Use["MLPBNN"]          = 0; // Recommended ANN with BFGS training method and bayesian regulator
   Use["CFMlpANN"]        = 0; // Depreciated ANN from ALEPH
   Use["TMlpANN"]         = 0; // ROOT's own ANN
   Use["DNN_CPU"] = 1;         // CUDA-accelerated DNN training.
   Use["DNN_GPU"] = 0;         // Multi-core accelerated DNN.
   //
   // Support Vector Machine
   Use["SVM"]             = 1;
   //
   // Boosted Decision Trees
   Use["BDT"]             = 1; // uses Adaptive Boost
   Use["BDTG"]            = 0; // uses Gradient Boost
   Use["BDTB"]            = 0; // uses Bagging
   Use["BDTD"]            = 0; // decorrelation + Adaptive Boost
   Use["BDTF"]            = 0; // allow usage of fisher discriminant for node splitting
   //
   // Friedman's RuleFit method, ie, an optimised series of cuts ("rules")
   Use["RuleFit"]         = 0;
   // ---------------------------------------------------------------
   Use["Plugin"]          = 0;
   Use["Category"]        = 0;
   Use["SVM_Gauss"]       = 0;
   Use["SVM_Poly"]        = 0;
   Use["SVM_Lin"]         = 0;

   std::cout << std::endl;
   std::cout << "==> Start TMVAClassificationApplication" << std::endl;

   // Select methods (don't look at this code - not of interest)
   if (myMethodList != "") {
      for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) it->second = 0;

      std::vector<TString> mlist = gTools().SplitString( myMethodList, ',' );
      for (UInt_t i=0; i<mlist.size(); i++) {
         std::string regMethod(mlist[i]);

         if (Use.find(regMethod) == Use.end()) {
            std::cout << "Method \"" << regMethod
                      << "\" not known in TMVA under this name. Choose among the following:" << std::endl;
            for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) {
               std::cout << it->first << " ";
            }
            std::cout << std::endl;
            return;
         }
         Use[regMethod] = 1;
      }
   }

   // --------------------------------------------------------------------------------------------------

   // Create the Reader object

   TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );

   // Create a set of variables and declare them to the reader
   // - the variable names MUST corresponds in name and type to those given in the weight file(s) used
   Float_t var1, var2;
   Float_t var3, var4;
   Float_t var5, var6;
   reader->AddVariable( "var1",           &var1 );
   reader->AddVariable( "var2",           &var2 );
   reader->AddVariable( "var3",           &var3 );
  // reader->AddVariable( "var4",           &var4 );
  // reader->AddVariable( "var5",           &var5 );
  // reader->AddVariable( "var6",           &var6 );

   // Spectator variables declared in the training have to be added to the reader, too
   Float_t spec1,spec2;
   reader->AddSpectator( "spec1 := var1*2",   &spec1 );
   reader->AddSpectator( "spec2 := var1*3",   &spec2 );

   Float_t Category_cat1, Category_cat2, Category_cat3;
   if (Use["Category"]){
      // Add artificial spectators for distinguishing categories
      reader->AddSpectator( "Category_cat1 := var3<=0",             &Category_cat1 );
      reader->AddSpectator( "Category_cat2 := (var3>0)",  &Category_cat2 );
      reader->AddSpectator( "Category_cat3 := (var3>0)", &Category_cat3 );
   }

   bool W_all=true;
   bool W_all_ecore=false;
   bool W_allN=false;
   bool W_antiproton=false;
   bool W_pion=false;
   bool W_Kion=false;

   bool data_single=false;
   bool data_embed=true;

   // Book the MVA methods
   TString dir;
  // if(W_all) dir = "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification1/dataset_allN_cutpt2/weights/";
  // if(W_all) dir = "dataset_allN_cutpt2_12/weights/";
    if(W_all) dir = "dataset_allN_cutpt2_12_embed/weights/";
  // if(W_all) dir = "dataset_allN_cutpt2_12_4vars/weights/";
  // if(W_all_ecore) dir = "dataset_allN_ecore/weights/";
   if(W_all_ecore) dir = "dataset_allN_ecore_cutpt2/weights/";
  // if(W_all) dir = "dataset_all_pt-ntpc_cut/weights/";
   if(W_allN) dir = "dataset_allN/weights/";
   if(W_antiproton) dir = "dataset_antiproton/weights/";
   if(W_pion) dir = "dataset_pion/weights/";
   if(W_Kion) dir = "dataset_Kion/weights/";
   TString prefix = "TMVAClassification";

   // Book method(s)
   for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) {
      if (it->second) {
         TString methodName = TString(it->first) + TString(" method");
         TString weightfile = dir + prefix + TString("_") + TString(it->first) + TString(".weights.xml");
         reader->BookMVA( methodName, weightfile );
      }
   }

   // Book output histograms
   UInt_t nbin = 100;
   TH1F *histCuts(0);//weihu
   TH1F *histLk(0);
   TH1F *histLkD(0);
   TH1F *histLkPCA(0);
   TH1F *histLkKDE(0);
   TH1F *histLkMIX(0);
   TH1F *histPD(0);
   TH1F *histPDD(0);
   TH1F *histPDPCA(0);
   TH1F *histPDEFoam(0);
   TH1F *histPDEFoamErr(0);
   TH1F *histPDEFoamSig(0);
   TH1F *histKNN(0);
   TH1F *histHm(0);
   TH1F *histFi(0);
   TH1F *histFiG(0);
   TH1F *histFiB(0);
   TH1F *histLD(0);
   TH1F *histNn(0);
   TH1F *histNnbfgs(0);
   TH1F *histNnbnn(0);
   TH1F *histNnC(0);
   TH1F *histNnT(0);
   TH1F *histBdt(0);
   TH1F *histBdtG(0);
   TH1F *histBdtB(0);
   TH1F *histBdtD(0);
   TH1F *histBdtF(0);
   TH1F *histRf(0);
   TH1F *histSVM(0);
   TH1F *histSVMG(0);
   TH1F *histSVMP(0);
   TH1F *histSVML(0);
   TH1F *histFDAMT(0);
   TH1F *histFDAGA(0);
   TH1F *histCat(0);
   TH1F *histPBdt(0);
   TH1F *histDnnGpu(0);
   TH1F *histDnnCpu(0);
   
   TH1F *h1electron_LD(0);
   TH1F *h1Sall_LD(0);
   TH1F *h1background_LD(0);
   TH1F *h1background_pion_LD(0);
   TH1F *h1background_antiproton_LD(0);
   TH1F *h1background_all_LD(0);
   TH1F *Hist_err_LD(0);
   TH1F *Hist_prob_LD(0);
   TH1F *Hist_rarity_LD(0);
   TH1F *Hist_Sig_LD(0);
   h1electron_LD = new TH1F( "h1electron_LD",    "h1electron_LD",    nbin, -2.0, 4.0 );//weihu
   h1Sall_LD= new TH1F( "h1Sall_LD",    "h1Sall_LD",    nbin, -2.0, 4.0 );//weihu
   h1background_LD = new TH1F( "h1background_LD",    "h1background_LD",    nbin, -2.0, 4.0 );//weihu
   h1background_pion_LD = new TH1F( "h1background_pion_LD",    "h1background_pion_LD",    nbin, -2.0, 4.0 );//weihu
   h1background_antiproton_LD = new TH1F( "h1background_antiproton_LD",    "h1background_antiproton_LD",    nbin, -2.0, 4.0 );//weihu
   h1background_all_LD = new TH1F( "h1background_all_LD",    "h1background_all_LD",    nbin, -2.0, 4.0 );//weihu
   Hist_err_LD = new TH1F( "Hist_err_LD",    "Hist_err_LD",    nbin, 0.0, 4.0 );//weihu
   Hist_prob_LD = new TH1F( "Hist_prob_LD",    "Hist_prob_LD",    nbin, 0.0, 1.0 );//weihu
   Hist_rarity_LD = new TH1F( "Hist_rarity_LD",    "Hist_rarity_LD",    nbin, 0.0, 1.0 );//weihu
   Hist_Sig_LD = new TH1F( "Hist_Sig_LD",    "Hist_Sig_LD",    nbin, -2.0, 4.0 );//weihu


   TH1F *h1electron_BDT(0);
   TH1F *h1Sall_BDT(0);
   TH1F *h1background_BDT(0);
   TH1F *h1background_pion_BDT(0);
   TH1F *h1background_antiproton_BDT(0);
   TH1F *h1background_all_BDT(0);
   h1electron_BDT = new TH1F( "h1electron_BDT",    "h1electron_BDT",    nbin, -1.0, 1.0 );//weihu
   h1Sall_BDT = new TH1F( "h1Sall_BDT",    "h1Sall_BDT",    nbin, -1.0, 1.0 );//weihu
   h1background_BDT = new TH1F( "h1background_BDT",    "h1background_BDT",    nbin, -1.0, 1.0 );//weihu
   h1background_pion_BDT = new TH1F( "h1background_pion_BDT",    "h1background_pion_BDT",    nbin, -1.0, 1.0 );//weihu
   h1background_antiproton_BDT = new TH1F( "h1background_antiproton_BDT",    "h1background_antiproton_BDT",    nbin, -1.0, 1.0 );//weihu
   h1background_all_BDT = new TH1F( "h1background_all_BDT",    "h1background_all_BDT",    nbin, -1.0, 1.0 );//weihu

   TH1F *h1electron_SVM(0);
   TH1F *h1Sall_SVM(0);
   TH1F *h1background_SVM(0);
   TH1F *h1background_pion_SVM(0);
   TH1F *h1background_antiproton_SVM(0);
   TH1F *h1background_all_SVM(0);
   h1electron_SVM = new TH1F( "h1electron_SVM",    "h1electron_SVM",    nbin, 0.0, 1.2 );//weihu
   h1Sall_SVM = new TH1F( "h1Sall_SVM",    "h1Sall_SVM",    nbin, 0.0, 1.2 );//weihu
   h1background_SVM = new TH1F( "h1background_SVM",    "h1background_SVM",    nbin, 0.0, 1.2 );//weihu
   h1background_pion_SVM = new TH1F( "h1background_pion_SVM",    "h1background_pion_SVM",    nbin, 0.0, 1.2 );//weihu
   h1background_antiproton_SVM = new TH1F( "h1background_antiproton_SVM",    "h1background_antiproton_SVM",    nbin, 0.0, 1.2 );//weihu
   h1background_all_SVM = new TH1F( "h1background_all_SVM",    "h1background_all_SVM",    nbin, 0.0, 1.2 );//weihu

   TH1F *h1electron_DNN_CPU(0);
   TH1F *h1Sall_DNN_CPU(0);
   TH1F *h1background_DNN_CPU(0);
   TH1F *h1background_pion_DNN_CPU(0);
   TH1F *h1background_antiproton_DNN_CPU(0);
   TH1F *h1background_all_DNN_CPU(0);
   h1electron_DNN_CPU = new TH1F( "h1electron_DNN_CPU",    "h1electron_DNN_CPU",    nbin, -0.2, 1.2 );//weihu
   h1Sall_DNN_CPU = new TH1F( "h1Sall_DNN_CPU",    "h1Sall_DNN_CPU",    nbin, -0.2, 1.2 );//weihu
   h1background_DNN_CPU = new TH1F( "h1background_DNN_CPU",    "h1background_DNN_CPU",    nbin, -0.2, 1.2 );//weihu
   h1background_pion_DNN_CPU = new TH1F( "h1background_pion_DNN_CPU",    "h1background_pion_DNN_CPU",    nbin, -0.2, 1.2 );//weihu
   h1background_antiproton_DNN_CPU = new TH1F( "h1background_antiproton_DNN_CPU",    "h1background_antiproton_DNN_CPU",    nbin, -0.2, 1.2 );//weihu
   h1background_all_DNN_CPU = new TH1F( "h1background_all_DNN_CPU",    "h1background_all_DNN_CPU",    nbin, -0.2, 1.2 );//weihu

   TH1F *h1EOP(0);//E3x3/p
   TH1F *h1EOP_e(0);//E3x3/p
   TH1F *h1EOP_cut(0);//E3x3/p
   TH1F *h1EcOP(0);//Ecore/p
   h1EOP = new TH1F( "h1EOP",    "h1EOP",    nbin, 0.0, 5.0 );
   h1EOP_e = new TH1F( "h1EOP_e",    "h1EOP_e",    50, 0.0, 2.0 );
   h1EOP_cut = new TH1F( "h1EOP_cut",    "h1EOP_cut",    nbin, 0.0, 5.0 );
   h1EcOP = new TH1F( "h1EcOP",    "h1EcOP",    nbin, 0.0, 5.0 );
   
   TH1F *h1HOM(0);//inH3x3/E3x3
   TH1F *h1HOM_e(0);//inH3x3/E3x3
   TH1F *h1CEMCchi2(0); //CEMC cluster Chi2
   TH1F *h1CEMCchi2_e(0); //CEMC cluster Chi2
   h1HOM = new TH1F( "h1HOM",    "h1HOM",    nbin, 0.0, 5.0 );
   h1HOM_e = new TH1F( "h1HOM_e",    "h1HOM_e",    nbin, 0.0, 5.0 );
   h1CEMCchi2 = new TH1F( "h1CEMCchi2",    "h1CEMCchi2",    nbin, 0.0, 20.0 );
   h1CEMCchi2_e = new TH1F( "h1CEMCchi2_e",    "h1CEMCchi2_e",    nbin, 0.0, 20.0 );

   TH1F *h1pt(0);
   TH1F *h1pt_cut(0);
   h1pt = new TH1F( "h1pt",    "h1pt",    nbin, 0.0, 20.0 );
   h1pt_cut = new TH1F( "h1pt_cut",    "h1pt_cut",    nbin, 0.0, 20.0 );

   TH1F *h1flavor_1(0);
   TH1F *h1flavor_2(0);
   h1flavor_1 = new TH1F( "h1flavor_1",    "h1flavor_1",    3000, -3000.0, 3000.0 );
   h1flavor_2 = new TH1F( "h1flavor_2",    "h1flavor_2",    3000, -3000.0, 3000.0 );

   TH1F *h1var1_EOP_1(0);
   TH1F *h1var2_HOM_1(0);
   TH1F *h1var3_Chi2_1(0);
   TH1F *h1var1_EOP_2(0);
   TH1F *h1var2_HOM_2(0);
   TH1F *h1var3_Chi2_2(0);
   h1var1_EOP_1 = new TH1F( "h1var1_EOP_1",    "h1var1_EOP_1",    30, 0.0, 3.0 );
   h1var2_HOM_1 = new TH1F( "h1var2_HOM_1",    "h1var2_HOM_1",    30, 0.0, 3.0 );
   h1var3_Chi2_1 = new TH1F( "h1var3_Chi2_1",    "h1var3_Chi2_1",    100, 0.0, 10.0 );
   h1var1_EOP_2 = new TH1F( "h1var1_EOP_2",    "h1var1_EOP_2",    30, 0.0, 3.0 );
   h1var2_HOM_2 = new TH1F( "h1var2_HOM_2",    "h1var2_HOM_2",    30, 0.0, 3.0 );
   h1var3_Chi2_2 = new TH1F( "h1var3_Chi2_2",    "h1var3_Chi2_2",    100, 0.0, 10.0 );

   TH1F *h1_p_1(0);
   TH1F *h1_pt_1(0);
   TH1F *h1_Eemcal3x3_1(0);
   TH1F *h1_p_2(0);
   TH1F *h1_pt_2(0);
   TH1F *h1_Eemcal3x3_2(0);
   h1_p_1 = new TH1F( "h1_p_1",    "h1_p_1",    100, 1.5, 49.5 );
   h1_pt_1 = new TH1F( "h1_pt_1",    "h1_pt_1",    100, 1.5, 29.5 );
   h1_Eemcal3x3_1 = new TH1F( "h1_Eemcal3x3_1",    "h1_Eemcal3x3_1",    180, 1.5, 19.5 );
   h1_p_2 = new TH1F( "h1_p_2",    "h1_p_2",    100, 1.5, 49.5 );
   h1_pt_2 = new TH1F( "h1_pt_2",    "h1_pt_2",    100, 1.5, 29.5 );
   h1_Eemcal3x3_2 = new TH1F( "h1_Eemcal3x3_2",    "h1_Eemcal3x3_2",    180, 1.5, 19.5 );

   TH2F *h2_reponse_pt(0);
   TH2F *h2_reponse_EOP(0);
   TH2F *h2_reponse_HOM(0);
   TH2F *h2_reponse_chi2(0);
   h2_reponse_pt = new TH2F( "h2_reponse_pt",    "h2_reponse_pt",  50,-0.5,0.5, 100, 1.5, 12.5 );
   h2_reponse_EOP = new TH2F( "h2_reponse_EOP",    "h2_reponse_EOP",  50,-0.5,0.5, 40, 0.0, 4.0 );
   h2_reponse_HOM = new TH2F( "h2_reponse_HOM",    "h2_reponse_HOM",  50,-0.5,0.5, 100, 0.0, 1.0 );
   h2_reponse_chi2 = new TH2F( "h2_reponse_chi2",    "h2_reponse_chi2",  50,-0.5,0.5, 200, 0.0, 20.0 );
   
   
   
   if (Use["Cuts"])          histCuts    = new TH1F( "MVA_Cuts",    "MVA_Cuts",    nbin, -2, 4 );//weihu

   if (Use["Likelihood"])    histLk      = new TH1F( "MVA_Likelihood",    "MVA_Likelihood",    nbin, -1, 1 );
   if (Use["LikelihoodD"])   histLkD     = new TH1F( "MVA_LikelihoodD",   "MVA_LikelihoodD",   nbin, -1, 0.9999 );
   if (Use["LikelihoodPCA"]) histLkPCA   = new TH1F( "MVA_LikelihoodPCA", "MVA_LikelihoodPCA", nbin, -1, 1 );
   if (Use["LikelihoodKDE"]) histLkKDE   = new TH1F( "MVA_LikelihoodKDE", "MVA_LikelihoodKDE", nbin,  -0.00001, 0.99999 );
   if (Use["LikelihoodMIX"]) histLkMIX   = new TH1F( "MVA_LikelihoodMIX", "MVA_LikelihoodMIX", nbin,  0, 1 );
   if (Use["PDERS"])         histPD      = new TH1F( "MVA_PDERS",         "MVA_PDERS",         nbin,  0, 1 );
   if (Use["PDERSD"])        histPDD     = new TH1F( "MVA_PDERSD",        "MVA_PDERSD",        nbin,  0, 1 );
   if (Use["PDERSPCA"])      histPDPCA   = new TH1F( "MVA_PDERSPCA",      "MVA_PDERSPCA",      nbin,  0, 1 );
   if (Use["KNN"])           histKNN     = new TH1F( "MVA_KNN",           "MVA_KNN",           nbin,  0, 1 );
   if (Use["HMatrix"])       histHm      = new TH1F( "MVA_HMatrix",       "MVA_HMatrix",       nbin, -0.95, 1.55 );
   if (Use["Fisher"])        histFi      = new TH1F( "MVA_Fisher",        "MVA_Fisher",        nbin, -4, 4 );
   if (Use["FisherG"])       histFiG     = new TH1F( "MVA_FisherG",       "MVA_FisherG",       nbin, -1, 1 );
   if (Use["BoostedFisher"]) histFiB     = new TH1F( "MVA_BoostedFisher", "MVA_BoostedFisher", nbin, -2, 2 );
   if (Use["LD"])            histLD      = new TH1F( "MVA_LD",            "MVA_LD",            nbin, -2, 2 );
   if (Use["MLP"])           histNn      = new TH1F( "MVA_MLP",           "MVA_MLP",           nbin, -1.25, 1.5 );
   if (Use["MLPBFGS"])       histNnbfgs  = new TH1F( "MVA_MLPBFGS",       "MVA_MLPBFGS",       nbin, -1.25, 1.5 );
   if (Use["MLPBNN"])        histNnbnn   = new TH1F( "MVA_MLPBNN",        "MVA_MLPBNN",        nbin, -1.25, 1.5 );
   if (Use["CFMlpANN"])      histNnC     = new TH1F( "MVA_CFMlpANN",      "MVA_CFMlpANN",      nbin,  0, 1 );
   if (Use["TMlpANN"])       histNnT     = new TH1F( "MVA_TMlpANN",       "MVA_TMlpANN",       nbin, -1.3, 1.3 );
   if (Use["DNN_GPU"])       histDnnGpu  = new TH1F("MVA_DNN_GPU",        "MVA_DNN_GPU",       nbin, -0.1, 1.1);
   if (Use["DNN_CPU"])       histDnnCpu  = new TH1F("MVA_DNN_CPU",        "MVA_DNN_CPU",       nbin, -0.1, 1.1);
   if (Use["BDT"])           histBdt     = new TH1F( "MVA_BDT",           "MVA_BDT",           nbin, -0.8, 0.8 );
   if (Use["BDTG"])          histBdtG    = new TH1F( "MVA_BDTG",          "MVA_BDTG",          nbin, -1.0, 1.0 );
   if (Use["BDTB"])          histBdtB    = new TH1F( "MVA_BDTB",          "MVA_BDTB",          nbin, -1.0, 1.0 );
   if (Use["BDTD"])          histBdtD    = new TH1F( "MVA_BDTD",          "MVA_BDTD",          nbin, -0.8, 0.8 );
   if (Use["BDTF"])          histBdtF    = new TH1F( "MVA_BDTF",          "MVA_BDTF",          nbin, -1.0, 1.0 );
   if (Use["RuleFit"])       histRf      = new TH1F( "MVA_RuleFit",       "MVA_RuleFit",       nbin, -2.0, 2.0 );
   if (Use["SVM"])           histSVM     = new TH1F( "MVA_SVM",           "MVA_SVM",           nbin,  0.0, 1.0 );
   if (Use["SVM_Gauss"])     histSVMG    = new TH1F( "MVA_SVM_Gauss",     "MVA_SVM_Gauss",     nbin,  0.0, 1.0 );
   if (Use["SVM_Poly"])      histSVMP    = new TH1F( "MVA_SVM_Poly",      "MVA_SVM_Poly",      nbin,  0.0, 1.0 );
   if (Use["SVM_Lin"])       histSVML    = new TH1F( "MVA_SVM_Lin",       "MVA_SVM_Lin",       nbin,  0.0, 1.0 );
   if (Use["FDA_MT"])        histFDAMT   = new TH1F( "MVA_FDA_MT",        "MVA_FDA_MT",        nbin, -2.0, 3.0 );
   if (Use["FDA_GA"])        histFDAGA   = new TH1F( "MVA_FDA_GA",        "MVA_FDA_GA",        nbin, -2.0, 3.0 );
   if (Use["Category"])      histCat     = new TH1F( "MVA_Category",      "MVA_Category",      nbin, -2., 2. );
   if (Use["Plugin"])        histPBdt    = new TH1F( "MVA_PBDT",          "MVA_BDT",           nbin, -0.8, 0.8 );

   // PDEFoam also returns per-event error, fill in histogram, and also fill significance
   if (Use["PDEFoam"]) {
      histPDEFoam    = new TH1F( "MVA_PDEFoam",       "MVA_PDEFoam",              nbin,  0, 1 );
      histPDEFoamErr = new TH1F( "MVA_PDEFoamErr",    "MVA_PDEFoam error",        nbin,  0, 1 );
      histPDEFoamSig = new TH1F( "MVA_PDEFoamSig",    "MVA_PDEFoam significance", nbin,  0, 10 );
   }

   // Book example histogram for probability (the other methods are done similarly)
   TH1F *probHistFi(0), *rarityHistFi(0);
   if (Use["Fisher"]) {
      probHistFi   = new TH1F( "MVA_Fisher_Proba",  "MVA_Fisher_Proba",  nbin, 0, 1 );
      rarityHistFi = new TH1F( "MVA_Fisher_Rarity", "MVA_Fisher_Rarity", nbin, 0, 1 );
   }

   // Prepare input tree (this must be replaced by your data source)
   // in this example, there is a toy tree with signal and one with background events
   // we'll later on use only the "signal" events for the test in this example.
   /*
   TFile *input(0);
   // TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data/tmva_class_example.root";
   TString fname = "/mnt/f/sPHSimu/sPHENIX/MVA/data/MVAdata_4vars.root";
   if (!gSystem->AccessPathName( fname )) {
      input = TFile::Open( fname ); // check if file in local directory exists
   }
   else {
      TFile::SetCacheFileDir(".");
      input = TFile::Open("http://root.cern.ch/files/tmva_class_example.root", "CACHEREAD"); // if not: download from ROOT server
   }
   if (!input) {
      std::cout << "ERROR: could not open data file" << std::endl;
      exit(1);
   }
   std::cout << "--- TMVAClassificationApp    : Using input file: " << input->GetName() << std::endl;

   // Event loop

   // Prepare the event tree
   // - Here the variable names have to corresponds to your tree
   // - You can use the same variables as above which is slightly faster,
   //   but of course you can use different ones and copy the values inside the event loop
   
   std::cout << "--- Select signal sample" << std::endl;
   TTree* theTree = (TTree*)input->Get("TreeSelectron");
  // TTree* theTree = (TTree*)input->Get("TreeBantiproton");
   Float_t userVar1, userVar2;
   theTree->SetBranchAddress( "var1", &userVar1 );
   theTree->SetBranchAddress( "var2", &userVar2 );
   theTree->SetBranchAddress( "var3", &var3 );
   theTree->SetBranchAddress( "var4", &var4 );
   */
  int Nfile;
  char input_file_embed[1000];
  char input_file_single[1000];
  const char *input_file[1000];
  const char *input_file_1[100];
  const char *input_file_2[100];
  const char *input_file_3[100];
  char input_file_tem[100];
  char input_file_tem1[100];
  char input_file_tem2[100];
  char input_file_tem3[100];
  char input_file_tem4[100];
  char input_file_tem5[100];
  char input_file_tem6[100];
  char input_file_tem7[100];
  for(int i;i<100;i++){
    sprintf(input_file_tem,"inatialization_%d",i);
    sprintf(input_file_tem1,"inatialization_%d",i);
    sprintf(input_file_tem2,"inatialization_%d",i);
    sprintf(input_file_tem3,"inatialization_%d",i);
    sprintf(input_file_tem4,"inatialization_%d",i);
    sprintf(input_file_tem5,"inatialization_%d",i);
    sprintf(input_file_tem6,"inatialization_%d",i);
    sprintf(input_file_tem7,"inatialization_%d",i);
  }

  // Efficiency calculator for cut method
   Int_t    nSelCutsGA = 0;
   Double_t effS       = 0.8;

   std::vector<Float_t> vecVar(4); // vector for EvaluateMVA tests

   
   TStopwatch sw;
   sw.Start();
   int N_raw=0, N_track=0, N_track_pt2=0;
   int Nelectron=0,Nelectron_cuts=0,Nelectron_BDT=0,Nelectron_SVM=0,nelectron_LD[10],nelectron_BDT[10],nelectron_SVM[10],nelectron_DNN_CPU[10];
   int NSall=0,nSall_LD[10],nSall_BDT[10],nSall_SVM[10],nSall_DNN_CPU[10];
   int Npion=0,npion_LD[10],npion_BDT[10],npion_SVM[10],npion_DNN_CPU[10];
   int Nantiproton=0,nantiproton_LD[10],nantiproton_BDT[10],nantiproton_SVM[10],nantiproton_DNN_CPU[10];
   int Nall=0,nall_LD[10],nall_BDT[10],nall_SVM[10],nall_DNN_CPU[10];
   float Ncut_LD[10],Ncut_BDT[10],Ncut_SVM[10],Ncut_DNN_CPU[10];
   float Npt[10],err_Npt[10],nall_SVM_pt[10],nall_BDT_pt[10],nall_cuts_pt[10],Nall_pt[10];
   float Nbimp[10],err_Nbimp[10],nall_SVM_bimp[10],nall_BDT_bimp[10],nall_cuts_bimp[10],Nall_bimp[10];

   float pt_point[10], N_electron_pt_cuts[10], NEID_electron_pt_cuts[10], N_electron_pt_BDT[10], NEID_electron_pt_BDT[10], N_electron_pt_SVM[10], NEID_electron_pt_SVM[10];

   for(int i=0;i<10;i++){
      nelectron_LD[i]=0;
      nSall_LD[i]=0;
      npion_LD[i]=0;
      nantiproton_LD[i]=0;
      nall_LD[i]=0;
      Ncut_LD[i]=0.0;

      nelectron_BDT[i]=0;
      nSall_BDT[i]=0;
      npion_BDT[i]=0;
      nantiproton_BDT[i]=0;
      nall_BDT[i]=0;
      Ncut_BDT[i]=0.0;

      nelectron_SVM[i]=0;
      nSall_SVM[i]=0;
      npion_SVM[i]=0;
      nantiproton_SVM[i]=0;
      nall_SVM[i]=0;
      Ncut_SVM[i]=0.0;

      nelectron_DNN_CPU[i]=0;
      nSall_DNN_CPU[i]=0;
      npion_DNN_CPU[i]=0;
      nantiproton_DNN_CPU[i]=0;
      nall_DNN_CPU[i]=0;
      Ncut_DNN_CPU[i]=0.0;

      Npt[i]=0.0;
      err_Npt[i]=0.0;
      Nall_pt[i]=0.0;
      nall_SVM_pt[i]=0.0;
      nall_BDT_pt[i]=0.0;
      nall_cuts_pt[i]=0.0;

      Nbimp[i]=0.0;
      err_Nbimp[i]=0.0;
      Nall_bimp[i]=0.0;
      nall_SVM_bimp[i]=0.0;
      nall_BDT_bimp[i]=0.0;
      nall_cuts_bimp[i]=0.0;

      pt_point[i]=0.0;
      N_electron_pt_cuts[i]=0.0;
      NEID_electron_pt_cuts[i]=0.0;
      N_electron_pt_BDT[i]=0.0;
      NEID_electron_pt_BDT[i]=0.0;
      N_electron_pt_SVM[i]=0.0;
      NEID_electron_pt_SVM[i]=0.0;
   }
   if(data_embed) Nfile=189;//189-old; 339-old+new
  // if(data_single) Nfile=24;
  // if(data_single) Nfile=481;
  // if(data_single) Nfile=481+72+67+67+67;
    if(data_single) Nfile=846;//846


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(data_embed){
        input_file[0] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_e-_embedHijing_50kHz_bkg_0_20fm-0000000004-00000_POSCOR_anaTutorial_50evt_20embed_e-.root";
        input_file[1] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_e-_embedHijing_50kHz_bkg_0_20fm-0000000004-00002_POSCOR_anaTutorial_50evt_20embed_e-.root";
        input_file[2] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_e-_embedHijing_50kHz_bkg_0_20fm-0000000004-00003_POSCOR_anaTutorial_50evt_20embed_e-.root";
        input_file[3] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_pi-_embedHijing_50kHz_bkg_0_20fm-0000000004-00004_POSCOR_anaTutorial_50evt_20embed_pi-.root";
        input_file[4] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_pi-_embedHijing_50kHz_bkg_0_20fm-0000000004-00005_POSCOR_anaTutorial_50evt_20embed_pi-.root";
        input_file[5] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_pi-_embedHijing_50kHz_bkg_0_20fm-0000000004-00006_POSCOR_anaTutorial_50evt_20embed_pi-.root";
        input_file[6] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_antiproton_embedHijing_50kHz_bkg_0_20fm-0000000004-00000_POSCOR_anaTutorial.root";
        input_file[7] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_antiproton_embedHijing_50kHz_bkg_0_20fm-0000000004-00001_POSCOR_anaTutorial.root";
        input_file[8] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_antiproton_embedHijing_50kHz_bkg_0_20fm-0000000004-00002_POSCOR_anaTutorial.root";
        input_file[9] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_K-_embedHijing_50kHz_bkg_0_20fm-0000000004-00000_POSCOR_anaTutorial.root";
        input_file[10] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_K-_embedHijing_50kHz_bkg_0_20fm-0000000004-00001_POSCOR_anaTutorial.root";
        input_file[11] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_embedHijing_50kHz_bkg_0_20fm-0000000004-00000_POSCOR_anaTutorial.root";
        input_file[12] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_embedHijing_50kHz_bkg_0_20fm-0000000004-00001_POSCOR_anaTutorial.root";
        input_file[13] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_embedHijing_50kHz_bkg_0_20fm-0000000004-00002_POSCOR_anaTutorial.root";
        input_file[14] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_embedHijing_50kHz_bkg_0_20fm-0000000004-00003_POSCOR_anaTutorial.root";
        input_file[15] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_embedHijing_50kHz_bkg_0_20fm-0000000004-00003_POSCOR_anaTutorial.root";
        input_file[16] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_embedHijing_50kHz_bkg_0_20fm-0000000004-00004_POSCOR_anaTutorial.root";
        input_file[17] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_embedHijing_50kHz_bkg_0_20fm-0000000004-00004_POSCOR_anaTutorial.root";
        input_file[18] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_embedHijing_50kHz_bkg_0_20fm-0000000004-00005_POSCOR_anaTutorial.root";

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        input_file[19] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-00_POSCOR_anaTutorial.root";
        input_file[20] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-01_POSCOR_anaTutorial.root";
        input_file[21] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-02_POSCOR_anaTutorial.root";
        input_file[22] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-03_POSCOR_anaTutorial.root";
        input_file[23] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-04_POSCOR_anaTutorial.root";
        input_file[24] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-05_POSCOR_anaTutorial.root";
        input_file[25] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-06_POSCOR_anaTutorial.root";
        input_file[26] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-07_POSCOR_anaTutorial.root";
        input_file[27] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-08_POSCOR_anaTutorial.root";
        input_file[28] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-09_POSCOR_anaTutorial.root";
        input_file[29] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-010_POSCOR_anaTutorial.root";
        input_file[30] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-011_POSCOR_anaTutorial.root";
        input_file[31] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-012_POSCOR_anaTutorial.root";
        input_file[32] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-013_POSCOR_anaTutorial.root";
        input_file[33] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-014_POSCOR_anaTutorial.root";
        input_file[34] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-015_POSCOR_anaTutorial.root";
        input_file[35] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-016_POSCOR_anaTutorial.root";
        input_file[36] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-017_POSCOR_anaTutorial.root";
        input_file[37] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-018_POSCOR_anaTutorial.root";
        input_file[38] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-019_POSCOR_anaTutorial.root";

        input_file[39] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-00_POSCOR_anaTutorial.root";
        input_file[40] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-01_POSCOR_anaTutorial.root";
        input_file[41] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-02_POSCOR_anaTutorial.root";
        input_file[42] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-03_POSCOR_anaTutorial.root";
        input_file[43] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-04_POSCOR_anaTutorial.root";
        input_file[44] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-05_POSCOR_anaTutorial.root";
        input_file[45] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-06_POSCOR_anaTutorial.root";
        input_file[46] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-07_POSCOR_anaTutorial.root";
        input_file[47] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-08_POSCOR_anaTutorial.root";
        input_file[48] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-09_POSCOR_anaTutorial.root";

        input_file[49] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-00_POSCOR_anaTutorial.root";
        input_file[50] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-01_POSCOR_anaTutorial.root";
        input_file[51] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-02_POSCOR_anaTutorial.root";
        input_file[52] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-03_POSCOR_anaTutorial.root";
        input_file[53] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-04_POSCOR_anaTutorial.root";
        input_file[54] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-05_POSCOR_anaTutorial.root";
        input_file[55] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-06_POSCOR_anaTutorial.root";
        input_file[56] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-07_POSCOR_anaTutorial.root";
        input_file[57] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-08_POSCOR_anaTutorial.root";
        input_file[58] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-09_POSCOR_anaTutorial.root";
        input_file[59] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-010_POSCOR_anaTutorial.root";
        input_file[60] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-011_POSCOR_anaTutorial.root";
        input_file[61] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-012_POSCOR_anaTutorial.root";
        input_file[62] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-013_POSCOR_anaTutorial.root";
        input_file[63] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-014_POSCOR_anaTutorial.root";
        input_file[64] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-015_POSCOR_anaTutorial.root";
        input_file[65] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-016_POSCOR_anaTutorial.root";
        input_file[66] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-017_POSCOR_anaTutorial.root";
        input_file[67] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-018_POSCOR_anaTutorial.root";
        input_file[68] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-019_POSCOR_anaTutorial.root";

        input_file[69] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-00_POSCOR_anaTutorial.root";
        input_file[70] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-01_POSCOR_anaTutorial.root";
        input_file[71] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-02_POSCOR_anaTutorial.root";
        input_file[72] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-03_POSCOR_anaTutorial.root";
        input_file[73] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-04_POSCOR_anaTutorial.root";
        input_file[74] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-05_POSCOR_anaTutorial.root";
        input_file[75] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-06_POSCOR_anaTutorial.root";
        input_file[76] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-07_POSCOR_anaTutorial.root";
        input_file[77] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-08_POSCOR_anaTutorial.root";
        input_file[78] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-09_POSCOR_anaTutorial.root";
        input_file[79] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-010_POSCOR_anaTutorial.root";
        input_file[80] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-011_POSCOR_anaTutorial.root";
        input_file[81] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-012_POSCOR_anaTutorial.root";
        input_file[82] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-013_POSCOR_anaTutorial.root";
        input_file[83] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-014_POSCOR_anaTutorial.root";
        input_file[84] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-015_POSCOR_anaTutorial.root";
        input_file[85] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-016_POSCOR_anaTutorial.root";
        input_file[86] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-017_POSCOR_anaTutorial.root";
        input_file[87] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-018_POSCOR_anaTutorial.root";
        input_file[88] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-019_POSCOR_anaTutorial.root";

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        input_file[89] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-00_POSCOR_anaTutorial.root";
        input_file[90] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-01_POSCOR_anaTutorial.root";
        input_file[91] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-02_POSCOR_anaTutorial.root";
        input_file[92] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-03_POSCOR_anaTutorial.root";
        input_file[93] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-04_POSCOR_anaTutorial.root";
        input_file[94] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-05_POSCOR_anaTutorial.root";
        input_file[95] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-06_POSCOR_anaTutorial.root";
        input_file[96] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-07_POSCOR_anaTutorial.root";
        input_file[97] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-08_POSCOR_anaTutorial.root";
        input_file[98] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-09_POSCOR_anaTutorial.root";
        input_file[99] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-010_POSCOR_anaTutorial.root";
        input_file[100] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-011_POSCOR_anaTutorial.root";
        input_file[101] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-012_POSCOR_anaTutorial.root";
        input_file[102] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-013_POSCOR_anaTutorial.root";
        input_file[103] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-014_POSCOR_anaTutorial.root";
        input_file[104] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-015_POSCOR_anaTutorial.root";
        input_file[105] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-016_POSCOR_anaTutorial.root";
        input_file[106] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-017_POSCOR_anaTutorial.root";
        input_file[107] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-018_POSCOR_anaTutorial.root";
        input_file[108] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-019_POSCOR_anaTutorial.root";
        input_file[109] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-020_POSCOR_anaTutorial.root";
        input_file[110] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-021_POSCOR_anaTutorial.root";
        input_file[111] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-022_POSCOR_anaTutorial.root";
        input_file[112] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-023_POSCOR_anaTutorial.root";
        input_file[113] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-024_POSCOR_anaTutorial.root";
        input_file[114] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-025_POSCOR_anaTutorial.root";
        input_file[115] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-026_POSCOR_anaTutorial.root";
        input_file[116] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-027_POSCOR_anaTutorial.root";
        input_file[117] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-028_POSCOR_anaTutorial.root";
        input_file[118] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-029_POSCOR_anaTutorial.root";
        input_file[119] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-030_POSCOR_anaTutorial.root";
        input_file[120] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-031_POSCOR_anaTutorial.root";
        input_file[121] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-032_POSCOR_anaTutorial.root";
        input_file[122] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-033_POSCOR_anaTutorial.root";
        input_file[123] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-034_POSCOR_anaTutorial.root";
        input_file[124] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-035_POSCOR_anaTutorial.root";
        input_file[125] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-036_POSCOR_anaTutorial.root";
        input_file[126] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-037_POSCOR_anaTutorial.root";
        input_file[127] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-038_POSCOR_anaTutorial.root";
        input_file[128] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-039_POSCOR_anaTutorial.root";
        input_file[129] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-040_POSCOR_anaTutorial.root";
        input_file[130] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-041_POSCOR_anaTutorial.root";
        input_file[131] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-042_POSCOR_anaTutorial.root";
        input_file[132] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-043_POSCOR_anaTutorial.root";
        input_file[133] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-044_POSCOR_anaTutorial.root";
        input_file[134] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-045_POSCOR_anaTutorial.root";
        input_file[135] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-046_POSCOR_anaTutorial.root";
        input_file[136] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-047_POSCOR_anaTutorial.root";
        input_file[137] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-048_POSCOR_anaTutorial.root";
        input_file[138] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-049_POSCOR_anaTutorial.root";
        input_file[139] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-050_POSCOR_anaTutorial.root";
        input_file[140] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-051_POSCOR_anaTutorial.root";
        input_file[141] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-052_POSCOR_anaTutorial.root";
        input_file[142] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-053_POSCOR_anaTutorial.root";
        input_file[143] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-054_POSCOR_anaTutorial.root";
        input_file[144] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-055_POSCOR_anaTutorial.root";
        input_file[145] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-056_POSCOR_anaTutorial.root";
        input_file[146] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-057_POSCOR_anaTutorial.root";
        input_file[147] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-058_POSCOR_anaTutorial.root";
        input_file[148] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-059_POSCOR_anaTutorial.root";
        input_file[149] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-060_POSCOR_anaTutorial.root";
        input_file[150] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-061_POSCOR_anaTutorial.root";
        input_file[151] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-062_POSCOR_anaTutorial.root";
        input_file[152] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-063_POSCOR_anaTutorial.root";
        input_file[153] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-064_POSCOR_anaTutorial.root";
        input_file[154] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-065_POSCOR_anaTutorial.root";
        input_file[155] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-066_POSCOR_anaTutorial.root";
        input_file[156] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-067_POSCOR_anaTutorial.root";
        input_file[157] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-068_POSCOR_anaTutorial.root";
        input_file[158] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-069_POSCOR_anaTutorial.root";
        input_file[159] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-070_POSCOR_anaTutorial.root";
        input_file[160] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-071_POSCOR_anaTutorial.root";
        input_file[161] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-072_POSCOR_anaTutorial.root";
        input_file[162] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-073_POSCOR_anaTutorial.root";
        input_file[163] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-074_POSCOR_anaTutorial.root";
        input_file[164] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-075_POSCOR_anaTutorial.root";
        input_file[165] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-076_POSCOR_anaTutorial.root";
        input_file[166] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-077_POSCOR_anaTutorial.root";
        input_file[167] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-078_POSCOR_anaTutorial.root";
        input_file[168] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-079_POSCOR_anaTutorial.root";

        input_file[169] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-020_POSCOR_anaTutorial.root";
        input_file[170] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-021_POSCOR_anaTutorial.root";
        input_file[171] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-022_POSCOR_anaTutorial.root";
        input_file[172] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-023_POSCOR_anaTutorial.root";
        input_file[173] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-024_POSCOR_anaTutorial.root";
        input_file[174] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-025_POSCOR_anaTutorial.root";
        input_file[175] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-026_POSCOR_anaTutorial.root";
        input_file[176] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-027_POSCOR_anaTutorial.root";
        input_file[177] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-028_POSCOR_anaTutorial.root";
        input_file[178] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-029_POSCOR_anaTutorial.root";
        input_file[179] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-030_POSCOR_anaTutorial.root";
        input_file[180] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-031_POSCOR_anaTutorial.root";
        input_file[181] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-032_POSCOR_anaTutorial.root";
        input_file[182] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-033_POSCOR_anaTutorial.root";
        input_file[183] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-034_POSCOR_anaTutorial.root";
        input_file[184] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-035_POSCOR_anaTutorial.root";
        input_file[185] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-036_POSCOR_anaTutorial.root";
        input_file[186] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-037_POSCOR_anaTutorial.root";
        input_file[187] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-038_POSCOR_anaTutorial.root";
        input_file[188] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-039_POSCOR_anaTutorial.root";
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        input_file[189] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_040_anaTutorial.root";
        input_file[190] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_041_anaTutorial.root";
        input_file[191] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_042_anaTutorial.root";
        input_file[192] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_043_anaTutorial.root";
        input_file[193] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_044_anaTutorial.root";
        input_file[194] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_045_anaTutorial.root";
        input_file[195] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_046_anaTutorial.root";
        input_file[196] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_047_anaTutorial.root";
        input_file[197] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_048_anaTutorial.root";
        input_file[198] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_049_anaTutorial.root";
        input_file[199] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_050_anaTutorial.root";
        input_file[200] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_051_anaTutorial.root";
        input_file[201] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_052_anaTutorial.root";
        input_file[202] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_053_anaTutorial.root";
        input_file[203] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_054_anaTutorial.root";
        input_file[204] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_055_anaTutorial.root";
        input_file[205] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_056_anaTutorial.root";
        input_file[206] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_057_anaTutorial.root";
        input_file[207] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_058_anaTutorial.root";
        input_file[208] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_059_anaTutorial.root";
        input_file[209] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_060_anaTutorial.root";
        input_file[210] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_061_anaTutorial.root";
        input_file[211] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_062_anaTutorial.root";
        input_file[212] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_063_anaTutorial.root";
        input_file[213] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20e-_bimp_embedHijing_pileup_0_20fm-0_064_anaTutorial.root";

        input_file[214] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_010_anaTutorial.root";
        input_file[215] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_011_anaTutorial.root";
        input_file[216] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_012_anaTutorial.root";
        input_file[217] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_013_anaTutorial.root";
        input_file[218] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_014_anaTutorial.root";
        input_file[219] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_015_anaTutorial.root";
        input_file[220] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_016_anaTutorial.root";
        input_file[221] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_017_anaTutorial.root";
        input_file[222] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_018_anaTutorial.root";
        input_file[223] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_034_anaTutorial.root";
        input_file[224] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_019_anaTutorial.root";
        input_file[225] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_020_anaTutorial.root";
        input_file[226] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_021_anaTutorial.root";
        input_file[227] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_022_anaTutorial.root";
        input_file[228] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_023_anaTutorial.root";
        input_file[229] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_024_anaTutorial.root";
        input_file[230] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_025_anaTutorial.root";
        input_file[231] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_026_anaTutorial.root";
        input_file[232] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_027_anaTutorial.root";
        input_file[233] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_028_anaTutorial.root";
        input_file[234] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_029_anaTutorial.root";
        input_file[235] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_030_anaTutorial.root";
        input_file[236] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_031_anaTutorial.root";
        input_file[237] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_032_anaTutorial.root";
        input_file[238] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20pi-_bimp_embedHijing_pileup_0_20fm-0_033_anaTutorial.root";

        input_file[239] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_020_anaTutorial.root";
        input_file[240] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_021_anaTutorial.root";
        input_file[241] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_022_anaTutorial.root";
        input_file[242] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_023_anaTutorial.root";
        input_file[243] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_024_anaTutorial.root";
        input_file[244] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_025_anaTutorial.root";
        input_file[245] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_026_anaTutorial.root";
        input_file[246] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_027_anaTutorial.root";
        input_file[247] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_028_anaTutorial.root";
        input_file[248] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_029_anaTutorial.root";
        input_file[249] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_030_anaTutorial.root";
        input_file[250] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_031_anaTutorial.root";
        input_file[251] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_032_anaTutorial.root";
        input_file[252] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_033_anaTutorial.root";
        input_file[253] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_034_anaTutorial.root";
        input_file[254] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_035_anaTutorial.root";
        input_file[255] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_036_anaTutorial.root";
        input_file[256] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_037_anaTutorial.root";
        input_file[257] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_038_anaTutorial.root";
        input_file[258] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_039_anaTutorial.root";
        input_file[259] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_040_anaTutorial.root";
        input_file[260] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_041_anaTutorial.root";
        input_file[261] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_042_anaTutorial.root";
        input_file[262] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_043_anaTutorial.root";
        input_file[263] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_044_anaTutorial.root";
        input_file[264] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_045_anaTutorial.root";
        input_file[265] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_046_anaTutorial.root";
        input_file[266] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_047_anaTutorial.root";
        input_file[267] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_048_anaTutorial.root";
        input_file[268] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20K-_bimp_embedHijing_pileup_0_20fm-0_049_anaTutorial.root";

        input_file[269] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_020_anaTutorial.root";
        input_file[270] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_021_anaTutorial.root";
        input_file[271] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_022_anaTutorial.root";
        input_file[272] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_023_anaTutorial.root";
        input_file[273] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_024_anaTutorial.root";
        input_file[274] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_025_anaTutorial.root";
        input_file[275] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_026_anaTutorial.root";
        input_file[276] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_027_anaTutorial.root";
        input_file[277] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_028_anaTutorial.root";
        input_file[278] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_029_anaTutorial.root";
        input_file[279] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_030_anaTutorial.root";
        input_file[280] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_031_anaTutorial.root";
        input_file[281] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_032_anaTutorial.root";
        input_file[282] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_033_anaTutorial.root";
        input_file[283] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_034_anaTutorial.root";
        input_file[284] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_035_anaTutorial.root";
        input_file[285] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_036_anaTutorial.root";
        input_file[286] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_037_anaTutorial.root";
        input_file[287] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_038_anaTutorial.root";
        input_file[288] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_039_anaTutorial.root";
        input_file[289] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_040_anaTutorial.root";
        input_file[290] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_041_anaTutorial.root";
        input_file[291] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_042_anaTutorial.root";
        input_file[292] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_043_anaTutorial.root";
        input_file[293] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_044_anaTutorial.root";
        input_file[294] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_045_anaTutorial.root";
        input_file[295] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_046_anaTutorial.root";
        input_file[296] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_047_anaTutorial.root";
        input_file[297] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_048_anaTutorial.root";
        input_file[298] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_20antiproton_bimp_embedHijing_pileup_0_20fm-0_049_anaTutorial.root";

        input_file[299] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_080_anaTutorial.root";
        input_file[300] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_081_anaTutorial.root";
        input_file[301] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_082_anaTutorial.root";
        input_file[302] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_083_anaTutorial.root";
        input_file[303] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_084_anaTutorial.root";
        input_file[304] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_085_anaTutorial.root";
        input_file[305] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_086_anaTutorial.root";
        input_file[306] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_087_anaTutorial.root";
        input_file[307] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_088_anaTutorial.root";
        input_file[308] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_089_anaTutorial.root";
        input_file[309] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_090_anaTutorial.root";
        input_file[310] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_091_anaTutorial.root";
        input_file[311] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_092_anaTutorial.root";
        input_file[312] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_093_anaTutorial.root";
        input_file[313] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_094_anaTutorial.root";
        input_file[314] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_095_anaTutorial.root";
        input_file[315] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_096_anaTutorial.root";
        input_file[316] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_097_anaTutorial.root";
        input_file[317] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_098_anaTutorial.root";
        input_file[318] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_20fm-0_099_anaTutorial.root";

        input_file[319] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_00_anaTutorial.root";
        input_file[320] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_01_anaTutorial.root";
        input_file[321] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_02_anaTutorial.root";
        input_file[322] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_03_anaTutorial.root";
        input_file[323] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_04_anaTutorial.root";
        input_file[324] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_05_anaTutorial.root";
        input_file[325] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_06_anaTutorial.root";
        input_file[326] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_07_anaTutorial.root";
        input_file[327] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_08_anaTutorial.root";
        input_file[328] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_09_anaTutorial.root";
        input_file[329] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_010_anaTutorial.root";
        input_file[330] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_011_anaTutorial.root";
        input_file[331] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_012_anaTutorial.root";
        input_file[332] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_013_anaTutorial.root";
        input_file[333] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_014_anaTutorial.root";
        input_file[334] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_015_anaTutorial.root";
        input_file[335] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_016_anaTutorial.root";
        input_file[336] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_017_anaTutorial.root";
        input_file[337] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_018_anaTutorial.root";
        input_file[338] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_Hijing_pileup_0_4p88fm-0_019_anaTutorial.root";

        
    }
    if(data_single){

    input_file[0] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_0_POSCOR.root_anaTutorial.root";//without truthflavor
    input_file[1] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_01_POSCOR_anaTutorial.root";//without truthflavor
    input_file[2] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_02_POSCOR_anaTutorial.root";//without truthflavor
    input_file[3] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_3_POSCOR_anaTutorial.root";//without truthflavor
    input_file[4] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_04_POSCOR_anaTutorial.root";//without truthflavor
    input_file[5] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_05_POSCOR_anaTutorial.root";//without truthflavor
    input_file[6] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_0_POSCOR_anaTutorial.root";//without truthflavor
    input_file[7] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_1_POSCOR_anaTutorial.root";//without truthflavor
    input_file[8] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_2_POSCOR_anaTutorial.root";//without truthflavor
    input_file[9] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_3_POSCOR_anaTutorial.root";//without truthflavor
    input_file[10] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_4_POSCOR_anaTutorial.root";//without truthflavor
    input_file[11] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_5_POSCOR_anaTutorial.root";//without truthflavor
    input_file[12] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_6_POSCOR_anaTutorial.root";//without truthflavor
    input_file[13] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_7_POSCOR_anaTutorial.root";
    input_file[14] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_8_POSCOR_anaTutorial.root";
    input_file[15] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_9_POSCOR_anaTutorial.root";
    input_file[16] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_10_POSCOR_anaTutorial.root";
    input_file[17] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_11_POSCOR_anaTutorial.root";
    input_file[18] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_12_POSCOR_anaTutorial.root";
    input_file[19] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_13_POSCOR_anaTutorial.root";
    input_file[20] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_14_POSCOR_anaTutorial.root";
    input_file[21] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_15_POSCOR_anaTutorial.root";
    input_file[22] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_16_POSCOR_anaTutorial.root";
    input_file[23] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_17_POSCOR_anaTutorial.root";
    input_file[24] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_18_POSCOR_anaTutorial.root";
    input_file[25] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_19_POSCOR_anaTutorial.root";
    input_file[26] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_200_20_POSCOR_anaTutorial.root";
    input_file[27] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_020_POSCOR_anaTutorial.root";
    input_file[28] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_021_POSCOR_anaTutorial.root";
    input_file[29] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_022_POSCOR_anaTutorial.root";
    input_file[30] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_023_POSCOR_anaTutorial.root";
    input_file[31] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_024_POSCOR_anaTutorial.root";
    input_file[32] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_025_POSCOR_anaTutorial.root";
    input_file[33] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_026_POSCOR_anaTutorial.root";
    input_file[34] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_027_POSCOR_anaTutorial.root";
    input_file[35] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_028_POSCOR_anaTutorial.root";
    input_file[36] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_030_POSCOR_anaTutorial.root";
    input_file[37] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_031_POSCOR_anaTutorial.root";
    input_file[38] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_032_POSCOR_anaTutorial.root";
    input_file[39] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_033_POSCOR_anaTutorial.root";
    input_file[40] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_034_POSCOR_anaTutorial.root";
    input_file[41] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_035_POSCOR_anaTutorial.root";
    input_file[42] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_036_POSCOR_anaTutorial.root";
    input_file[43] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_037_POSCOR_anaTutorial.root";
    input_file[44] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_038_POSCOR_anaTutorial.root";
    input_file[45] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_040_POSCOR_anaTutorial.root";
    input_file[46] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_041_POSCOR_anaTutorial.root";

    input_file[448] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_042_POSCOR_anaTutorial.root";
    input_file[449] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_044_POSCOR_anaTutorial.root";
    input_file[450] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_045_POSCOR_anaTutorial.root";
    input_file[451] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_048_POSCOR_anaTutorial.root";
    input_file[452] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_049_POSCOR_anaTutorial.root";
    input_file[453] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_051_POSCOR_anaTutorial.root";
    input_file[454] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_053_POSCOR_anaTutorial.root";
    input_file[455] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_054_POSCOR_anaTutorial.root";
    input_file[456] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_055_POSCOR_anaTutorial.root";
    input_file[457] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_056_POSCOR_anaTutorial.root";
    input_file[458] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_057_POSCOR_anaTutorial.root";
    input_file[459] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_058_POSCOR_anaTutorial.root";
    input_file[460] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_059_POSCOR_anaTutorial.root";
    input_file[461] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_060_POSCOR_anaTutorial.root";
    input_file[462] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_061_POSCOR_anaTutorial.root";
    input_file[463] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_062_POSCOR_anaTutorial.root";
    input_file[464] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_063_POSCOR_anaTutorial.root";
    input_file[465] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_064_POSCOR_anaTutorial.root";

    //N_e+=6600
    input_file[47] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_00_POSCOR_anaTutorial.root";
    input_file[48] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_01_POSCOR_anaTutorial.root";
    input_file[49] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_02_POSCOR_anaTutorial.root";
    input_file[50] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_03_POSCOR_anaTutorial.root";
    input_file[51] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_04_POSCOR_anaTutorial.root";
    input_file[52] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_05_POSCOR_anaTutorial.root";
    input_file[53] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_06_POSCOR_anaTutorial.root";
    input_file[54] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_07_POSCOR_anaTutorial.root";
    input_file[55] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_08_POSCOR_anaTutorial.root";
    input_file[56] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_09_POSCOR_anaTutorial.root";
    input_file[57] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_010_POSCOR_anaTutorial.root";
    input_file[58] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_011_POSCOR_anaTutorial.root";
    input_file[59] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_012_POSCOR_anaTutorial.root";
    input_file[60] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_013_POSCOR_anaTutorial.root";
    input_file[61] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_015_POSCOR_anaTutorial.root";
    input_file[62] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_016_POSCOR_anaTutorial.root";
    input_file[63] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_017_POSCOR_anaTutorial.root";
    input_file[64] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_018_POSCOR_anaTutorial.root";
    input_file[65] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_019_POSCOR_anaTutorial.root";
    input_file[66] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_020_POSCOR_anaTutorial.root";
    input_file[67] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_021_POSCOR_anaTutorial.root";
    input_file[68] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_022_POSCOR_anaTutorial.root";
    input_file[69] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_023_POSCOR_anaTutorial.root";
    input_file[70] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_024_POSCOR_anaTutorial.root";
    input_file[71] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_025_POSCOR_anaTutorial.root";
    input_file[72] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_026_POSCOR_anaTutorial.root";
    input_file[73] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_027_POSCOR_anaTutorial.root";
    input_file[74] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_028_POSCOR_anaTutorial.root";
    input_file[75] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_029_POSCOR_anaTutorial.root";
    input_file[76] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_031_POSCOR_anaTutorial.root";
    input_file[77] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_032_POSCOR_anaTutorial.root";
    input_file[78] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_033_POSCOR_anaTutorial.root";
    input_file[79] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_036_POSCOR_anaTutorial.root";
    input_file[80] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_037_POSCOR_anaTutorial.root";
    input_file[81] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_040_POSCOR_anaTutorial.root";
    input_file[82] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_041_POSCOR_anaTutorial.root";
    input_file[83] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_042_POSCOR_anaTutorial.root";
    input_file[84] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_043_POSCOR_anaTutorial.root";
    input_file[85] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_044_POSCOR_anaTutorial.root";
    input_file[86] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_045_POSCOR_anaTutorial.root";
    input_file[87] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_046_POSCOR_anaTutorial.root";
    input_file[88] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_047_POSCOR_anaTutorial.root";
    input_file[89] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_048_POSCOR_anaTutorial.root";
    input_file[90] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_049_POSCOR_anaTutorial.root";
    input_file[91] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_050_POSCOR_anaTutorial.root";
    input_file[92] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_051_POSCOR_anaTutorial.root";
    input_file[93] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_052_POSCOR_anaTutorial.root";
    input_file[94] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_053_POSCOR_anaTutorial.root";
    input_file[95] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_054_POSCOR_anaTutorial.root";
    input_file[96] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_055_POSCOR_anaTutorial.root";
    input_file[97] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_056_POSCOR_anaTutorial.root";
    input_file[98] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_057_POSCOR_anaTutorial.root";
    input_file[99] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_058_POSCOR_anaTutorial.root";
    input_file[100] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_059_POSCOR_anaTutorial.root";
    input_file[101] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_060_POSCOR_anaTutorial.root";
    input_file[102] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_061_POSCOR_anaTutorial.root";
    input_file[103] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_063_POSCOR_anaTutorial.root";
    input_file[104] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_064_POSCOR_anaTutorial.root";
    input_file[105] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_065_POSCOR_anaTutorial.root";
    input_file[106] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_066_POSCOR_anaTutorial.root";
    input_file[107] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_069_POSCOR_anaTutorial.root";
    input_file[108] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_070_POSCOR_anaTutorial.root";
    input_file[109] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_071_POSCOR_anaTutorial.root";
    input_file[110] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_072_POSCOR_anaTutorial.root";
    input_file[111] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_073_POSCOR_anaTutorial.root";
    input_file[112] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_074_POSCOR_anaTutorial.root";

    input_file[466] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_075_POSCOR_anaTutorial.root";
    input_file[467] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_076_POSCOR_anaTutorial.root";
    input_file[468] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_077_POSCOR_anaTutorial.root";
    input_file[469] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_078_POSCOR_anaTutorial.root";
    input_file[470] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_079_POSCOR_anaTutorial.root";
    input_file[471] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_080_POSCOR_anaTutorial.root";
    input_file[472] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_081_POSCOR_anaTutorial.root";
    input_file[473] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_082_POSCOR_anaTutorial.root";
    input_file[474] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_083_POSCOR_anaTutorial.root";
    input_file[475] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_084_POSCOR_anaTutorial.root";
    input_file[476] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_085_POSCOR_anaTutorial.root";
    input_file[477] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_086_POSCOR_anaTutorial.root";
    input_file[478] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_087_POSCOR_anaTutorial.root";
    input_file[479] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_088_POSCOR_anaTutorial.root";
    input_file[480] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e+_eta0-1.1_0-20GeV_100_089_POSCOR_anaTutorial.root";

    //N_antiproton=4000+3400=7400
    input_file[113] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_0_POSCOR_anaTutorial.root";//without truthflavor
    input_file[117] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_1_POSCOR_anaTutorial.root";//without truthflavor
    input_file[121] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_2_POSCOR_anaTutorial.root";//without truthflavor
    input_file[125] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_3_POSCOR_anaTutorial.root";
    input_file[129] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_4_POSCOR_anaTutorial.root";
    input_file[133] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_5_POSCOR_anaTutorial.root";
    input_file[137] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_6_POSCOR_anaTutorial.root";
    input_file[141] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_7_POSCOR_anaTutorial.root";
    input_file[145] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_8_POSCOR_anaTutorial.root";
    input_file[149] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_9_POSCOR_anaTutorial.root";
    input_file[153] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_10_POSCOR_anaTutorial.root";
    input_file[157] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_11_POSCOR_anaTutorial.root";
    input_file[161] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_12_POSCOR_anaTutorial.root";
    input_file[165] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_13_POSCOR_anaTutorial.root";
    input_file[169] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_14_POSCOR_anaTutorial.root";
    input_file[173] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_15_POSCOR_anaTutorial.root";
    input_file[177] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_16_POSCOR_anaTutorial.root";
    input_file[181] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_17_POSCOR_anaTutorial.root";
    input_file[185] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_18_POSCOR_anaTutorial.root";
    input_file[189] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_19_POSCOR_anaTutorial.root";
    input_file[193] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_20_POSCOR_anaTutorial.root";
    input_file[198] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_21_POSCOR_anaTutorial.root";
    input_file[203] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_200_22_POSCOR_anaTutorial.root";
    input_file[208] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_23_POSCOR_anaTutorial.root";
    input_file[211] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_24_POSCOR_anaTutorial.root";
    input_file[214] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_25_POSCOR_anaTutorial.root";
    input_file[217] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_26_POSCOR_anaTutorial.root";
    input_file[220] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_27_POSCOR_anaTutorial.root";
    input_file[223] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_28_POSCOR_anaTutorial.root";
    input_file[226] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_29_POSCOR_anaTutorial.root";
    input_file[229] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_30_POSCOR_anaTutorial.root";
    input_file[232] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_31_POSCOR_anaTutorial.root";
    input_file[235] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_32_POSCOR_anaTutorial.root";
    input_file[238] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_33_POSCOR_anaTutorial.root";
    input_file[241] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_34_POSCOR_anaTutorial.root";
    input_file[244] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_35_POSCOR_anaTutorial.root";
    input_file[247] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_36_POSCOR_anaTutorial.root";
    input_file[250] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_37_POSCOR_anaTutorial.root";
    input_file[253] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_38_POSCOR_anaTutorial.root";
    input_file[256] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_39_POSCOR_anaTutorial.root";
    input_file[258] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_40_POSCOR_anaTutorial.root";
    input_file[260] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_41_POSCOR_anaTutorial.root";
    input_file[262] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_42_POSCOR_anaTutorial.root";
    input_file[264] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_43_POSCOR_anaTutorial.root";
    input_file[266] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_44_POSCOR_anaTutorial.root";
    input_file[268] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_45_POSCOR_anaTutorial.root";
    input_file[269] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_46_POSCOR_anaTutorial.root";
    input_file[270] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_47_POSCOR_anaTutorial.root";
    input_file[271] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_48_POSCOR_anaTutorial.root";
    input_file[272] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_49_POSCOR_anaTutorial.root";
    input_file[273] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_0-20GeV_100_50_POSCOR_anaTutorial.root";//273 -


    //N_pi-=3000+3000=6000
    input_file[114] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_0_POSCOR_anaTutorial.root";
    input_file[118] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_1_POSCOR_anaTutorial.root";
    input_file[122] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_2_POSCOR_anaTutorial.root";
    input_file[126] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_3_POSCOR_anaTutorial.root";
    input_file[130] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_4_POSCOR_anaTutorial.root";
    input_file[134] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_5_POSCOR_anaTutorial.root";
    input_file[138] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_6_POSCOR_anaTutorial.root";
    input_file[142] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_7_POSCOR_anaTutorial.root";
    input_file[146] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_8_POSCOR_anaTutorial.root";
    input_file[150] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_9_POSCOR_anaTutorial.root";
    input_file[154] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_10_POSCOR_anaTutorial.root";
    input_file[158] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_11_POSCOR_anaTutorial.root";
    input_file[162] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_12_POSCOR_anaTutorial.root";
    input_file[166] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_13_POSCOR_anaTutorial.root";
    input_file[170] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_14_POSCOR_anaTutorial.root";
    input_file[174] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_15_POSCOR_anaTutorial.root";
    input_file[178] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_16_POSCOR_anaTutorial.root";
    input_file[182] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_17_POSCOR_anaTutorial.root";
    input_file[186] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_18_POSCOR_anaTutorial.root";
    input_file[190] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_200_19_POSCOR_anaTutorial.root";
    input_file[194] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_00_POSCOR_anaTutorial.root";
    input_file[195] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_01_POSCOR_anaTutorial.root";
    input_file[199] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_02_POSCOR_anaTutorial.root";
    input_file[200] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_03_POSCOR_anaTutorial.root";
    input_file[204] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_04_POSCOR_anaTutorial.root";
    input_file[205] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_05_POSCOR_anaTutorial.root";
    input_file[209] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_07_POSCOR_anaTutorial.root";
    input_file[212] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_09_POSCOR_anaTutorial.root";
    input_file[215] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_010_POSCOR_anaTutorial.root";
    input_file[218] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_011_POSCOR_anaTutorial.root";
    input_file[221] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_012_POSCOR_anaTutorial.root";
    input_file[224] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_013_POSCOR_anaTutorial.root";
    input_file[227] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_014_POSCOR_anaTutorial.root";
    input_file[230] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_015_POSCOR_anaTutorial.root";
    input_file[233] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_017_POSCOR_anaTutorial.root";
    input_file[236] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_018_POSCOR_anaTutorial.root";
    input_file[239] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_019_POSCOR_anaTutorial.root";
    input_file[242] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_021_POSCOR_anaTutorial.root";
    input_file[245] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_022_POSCOR_anaTutorial.root";
    input_file[248] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_023_POSCOR_anaTutorial.root";
    input_file[251] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_024_POSCOR_anaTutorial.root";
    input_file[254] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_025_POSCOR_anaTutorial.root";
    input_file[257] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_026_POSCOR_anaTutorial.root";
    input_file[259] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_028_POSCOR_anaTutorial.root";
    input_file[261] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_029_POSCOR_anaTutorial.root";
    input_file[263] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_030_POSCOR_anaTutorial.root";
    input_file[265] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_031_POSCOR_anaTutorial.root";
    input_file[267] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_0-20GeV_100_032_POSCOR_anaTutorial.root";


     //N_K-=6600
    input_file[115] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_00_POSCOR_anaTutorial.root";
    input_file[116] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_01_POSCOR_anaTutorial.root";
    input_file[119] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_02_POSCOR_anaTutorial.root";
    input_file[120] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_03_POSCOR_anaTutorial.root";
    input_file[123] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_04_POSCOR_anaTutorial.root";
    input_file[124] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_05_POSCOR_anaTutorial.root";
    input_file[127] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_06_POSCOR_anaTutorial.root";
    input_file[128] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_07_POSCOR_anaTutorial.root";
    input_file[131] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_08_POSCOR_anaTutorial.root";
    input_file[132] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_09_POSCOR_anaTutorial.root";
    input_file[135] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_010_POSCOR_anaTutorial.root";
    input_file[136] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_012_POSCOR_anaTutorial.root";
    input_file[139] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_013_POSCOR_anaTutorial.root";
    input_file[140] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_014_POSCOR_anaTutorial.root";
    input_file[143] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_015_POSCOR_anaTutorial.root";
    input_file[144] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_016_POSCOR_anaTutorial.root";
    input_file[147] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_017_POSCOR_anaTutorial.root";
    input_file[148] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_018_POSCOR_anaTutorial.root";
    input_file[151] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_019_POSCOR_anaTutorial.root";
    input_file[152] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_021_POSCOR_anaTutorial.root";
    input_file[155] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_022_POSCOR_anaTutorial.root";
    input_file[156] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_023_POSCOR_anaTutorial.root";
    input_file[159] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_024_POSCOR_anaTutorial.root";
    input_file[160] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_025_POSCOR_anaTutorial.root";
    input_file[163] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_027_POSCOR_anaTutorial.root";
    input_file[164] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_028_POSCOR_anaTutorial.root";
    input_file[167] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_030_POSCOR_anaTutorial.root";
    input_file[168] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_031_POSCOR_anaTutorial.root";
    input_file[171] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_032_POSCOR_anaTutorial.root";
    input_file[172] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_033_POSCOR_anaTutorial.root";
    input_file[175] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_035_POSCOR_anaTutorial.root";
    input_file[176] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_036_POSCOR_anaTutorial.root";
    input_file[179] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_037_POSCOR_anaTutorial.root";
    input_file[180] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_038_POSCOR_anaTutorial.root";
    input_file[183] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_039_POSCOR_anaTutorial.root";
    input_file[184] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_040_POSCOR_anaTutorial.root";
    input_file[187] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_041_POSCOR_anaTutorial.root";
    input_file[188] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_042_POSCOR_anaTutorial.root";
    input_file[191] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_043_POSCOR_anaTutorial.root";
    input_file[192] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_044_POSCOR_anaTutorial.root";
    input_file[196] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_045_POSCOR_anaTutorial.root";
    input_file[197] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_046_POSCOR_anaTutorial.root";
    input_file[201] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_047_POSCOR_anaTutorial.root";
    input_file[202] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_048_POSCOR_anaTutorial.root";
    input_file[206] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_049_POSCOR_anaTutorial.root";
    input_file[207] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_050_POSCOR_anaTutorial.root";
    input_file[210] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_052_POSCOR_anaTutorial.root";
    input_file[213] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_053_POSCOR_anaTutorial.root";
    input_file[216] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_054_POSCOR_anaTutorial.root";
    input_file[219] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_055_POSCOR_anaTutorial.root";
    input_file[222] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_056_POSCOR_anaTutorial.root";
    input_file[225] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_057_POSCOR_anaTutorial.root";
    input_file[228] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_059_POSCOR_anaTutorial.root";
    input_file[231] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_060_POSCOR_anaTutorial.root";
    input_file[234] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_061_POSCOR_anaTutorial.root";
    input_file[237] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_062_POSCOR_anaTutorial.root";
    input_file[240] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_063_POSCOR_anaTutorial.root";
    input_file[243] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_065_POSCOR_anaTutorial.root";
    input_file[246] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_066_POSCOR_anaTutorial.root";
    input_file[249] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_067_POSCOR_anaTutorial.root";
    input_file[252] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_069_POSCOR_anaTutorial.root";
    input_file[255] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_0-20GeV_100_070_POSCOR_anaTutorial.root";


     //N_proton=6700
    input_file[274] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_200_0_POSCOR_anaTutorial.root";//274+
    input_file[279] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_200_1_POSCOR_anaTutorial.root";
    input_file[284] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_200_2_POSCOR_anaTutorial.root";
    input_file[289] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_200_3_POSCOR_anaTutorial.root";
    input_file[294] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_200_4_POSCOR_anaTutorial.root";
    input_file[299] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_200_5_POSCOR_anaTutorial.root";
    input_file[304] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_200_6_POSCOR_anaTutorial.root";
    input_file[309] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_200_7_POSCOR_anaTutorial.root";
    input_file[314] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_200_8_POSCOR_anaTutorial.root";
    input_file[319] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_200_9_POSCOR_anaTutorial.root";
    input_file[324] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_01_POSCOR_anaTutorial.root";
    input_file[327] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_02_POSCOR_anaTutorial.root";
    input_file[330] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_03_POSCOR_anaTutorial.root";
    input_file[333] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_04_POSCOR_anaTutorial.root";
    input_file[336] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_05_POSCOR_anaTutorial.root";
    input_file[339] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_06_POSCOR_anaTutorial.root";
    input_file[342] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_07_POSCOR_anaTutorial.root";
    input_file[345] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_08_POSCOR_anaTutorial.root";
    input_file[348] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_09_POSCOR_anaTutorial.root";
    input_file[351] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_010_POSCOR_anaTutorial.root";
    input_file[354] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_011_POSCOR_anaTutorial.root";
    input_file[357] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_013_POSCOR_anaTutorial.root";
    input_file[360] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_014_POSCOR_anaTutorial.root";
    input_file[363] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_015_POSCOR_anaTutorial.root";
    input_file[369] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_016_POSCOR_anaTutorial.root";
    input_file[372] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_017_POSCOR_anaTutorial.root";
    input_file[375] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_018_POSCOR_anaTutorial.root";
    input_file[378] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_019_POSCOR_anaTutorial.root";
    input_file[381] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_020_POSCOR_anaTutorial.root";
    input_file[384] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_021_POSCOR_anaTutorial.root";
    input_file[387] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_022_POSCOR_anaTutorial.root";
    input_file[390] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_023_POSCOR_anaTutorial.root";
    input_file[393] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_024_POSCOR_anaTutorial.root";
    input_file[396] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_025_POSCOR_anaTutorial.root";
    input_file[399] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_026_POSCOR_anaTutorial.root";
    input_file[402] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_027_POSCOR_anaTutorial.root";
    input_file[405] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_028_POSCOR_anaTutorial.root";
    input_file[408] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_029_POSCOR_anaTutorial.root";
    input_file[411] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_030_POSCOR_anaTutorial.root";
    input_file[414] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_031_POSCOR_anaTutorial.root";
    input_file[417] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_032_POSCOR_anaTutorial.root";
    input_file[420] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_033_POSCOR_anaTutorial.root";
    input_file[423] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_034_POSCOR_anaTutorial.root";
    input_file[426] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_035_POSCOR_anaTutorial.root";
    input_file[429] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_036_POSCOR_anaTutorial.root";
    input_file[432] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_037_POSCOR_anaTutorial.root";
    input_file[435] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_038_POSCOR_anaTutorial.root";
   
    input_file[438] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_039_POSCOR_anaTutorial.root";
    input_file[439] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_040_POSCOR_anaTutorial.root";
    input_file[440] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_041_POSCOR_anaTutorial.root";
    input_file[441] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_042_POSCOR_anaTutorial.root";
    input_file[442] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_043_POSCOR_anaTutorial.root";
    input_file[443] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_044_POSCOR_anaTutorial.root";
    input_file[444] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_045_POSCOR_anaTutorial.root";
    input_file[445] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_046_POSCOR_anaTutorial.root";
    input_file[446] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_047_POSCOR_anaTutorial.root";
    input_file[447] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_proton_eta0-1.1_0-20GeV_100_048_POSCOR_anaTutorial.root";

    //N_pi+=5800
    input_file[275] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_00_POSCOR_anaTutorial.root";
    input_file[276] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_02_POSCOR_anaTutorial.root";
    input_file[280] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_03_POSCOR_anaTutorial.root";
    input_file[281] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_04_POSCOR_anaTutorial.root";
    input_file[285] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_05_POSCOR_anaTutorial.root";
    input_file[286] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_06_POSCOR_anaTutorial.root";
    input_file[290] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_07_POSCOR_anaTutorial.root";
    input_file[291] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_08_POSCOR_anaTutorial.root";
    input_file[295] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_09_POSCOR_anaTutorial.root";
    input_file[296] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_011_POSCOR_anaTutorial.root";
    input_file[300] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_012_POSCOR_anaTutorial.root";
    input_file[301] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_014_POSCOR_anaTutorial.root";
    input_file[305] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_015_POSCOR_anaTutorial.root";
    input_file[306] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_016_POSCOR_anaTutorial.root";
    input_file[310] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_017_POSCOR_anaTutorial.root";
    input_file[311] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_018_POSCOR_anaTutorial.root";
    input_file[315] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_019_POSCOR_anaTutorial.root";
    input_file[316] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_021_POSCOR_anaTutorial.root";
    input_file[320] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_022_POSCOR_anaTutorial.root";
    input_file[321] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_023_POSCOR_anaTutorial.root";
    input_file[325] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_024_POSCOR_anaTutorial.root";
    input_file[328] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_025_POSCOR_anaTutorial.root";
    input_file[331] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_026_POSCOR_anaTutorial.root";
    input_file[334] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_027_POSCOR_anaTutorial.root";
    input_file[337] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_028_POSCOR_anaTutorial.root";
    input_file[340] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_029_POSCOR_anaTutorial.root";
    input_file[343] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_030_POSCOR_anaTutorial.root";
    input_file[346] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_031_POSCOR_anaTutorial.root";
    input_file[349] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_032_POSCOR_anaTutorial.root";
    input_file[352] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_033_POSCOR_anaTutorial.root";
    input_file[355] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_034_POSCOR_anaTutorial.root";
    input_file[358] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_035_POSCOR_anaTutorial.root";
    input_file[361] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_036_POSCOR_anaTutorial.root";
    input_file[364] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_037_POSCOR_anaTutorial.root";
    input_file[367] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_038_POSCOR_anaTutorial.root";
    input_file[370] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_039_POSCOR_anaTutorial.root";
    input_file[373] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_040_POSCOR_anaTutorial.root";
    input_file[376] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_042_POSCOR_anaTutorial.root";
    input_file[379] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_043_POSCOR_anaTutorial.root";
    input_file[382] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_044_POSCOR_anaTutorial.root";
    input_file[385] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_045_POSCOR_anaTutorial.root";
    input_file[388] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_047_POSCOR_anaTutorial.root";
    input_file[391] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_048_POSCOR_anaTutorial.root";
    input_file[394] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_049_POSCOR_anaTutorial.root";
    input_file[397] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_050_POSCOR_anaTutorial.root";
    input_file[400] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_051_POSCOR_anaTutorial.root";
    input_file[403] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_052_POSCOR_anaTutorial.root";
    input_file[406] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_053_POSCOR_anaTutorial.root";
    input_file[409] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_054_POSCOR_anaTutorial.root";
    input_file[412] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_055_POSCOR_anaTutorial.root";
    input_file[415] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_056_POSCOR_anaTutorial.root";
    input_file[418] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_057_POSCOR_anaTutorial.root";
    input_file[421] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_058_POSCOR_anaTutorial.root";
    input_file[424] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_059_POSCOR_anaTutorial.root";
    input_file[427] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_060_POSCOR_anaTutorial.root";
    input_file[430] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_061_POSCOR_anaTutorial.root";
    input_file[433] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_01_POSCOR_anaTutorial.root";
    input_file[436] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi+_eta0-1.1_0-20GeV_100_010_POSCOR_anaTutorial.root";

   
    //N_K+=5800
    input_file[277] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_00_POSCOR_anaTutorial.root";
    input_file[278] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_01_POSCOR_anaTutorial.root";
    input_file[282] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_03_POSCOR_anaTutorial.root";
    input_file[283] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_04_POSCOR_anaTutorial.root";
    input_file[287] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_05_POSCOR_anaTutorial.root";
    input_file[288] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_06_POSCOR_anaTutorial.root";
    input_file[292] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_07_POSCOR_anaTutorial.root";
    input_file[293] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_08_POSCOR_anaTutorial.root";
    input_file[297] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_09_POSCOR_anaTutorial.root";
    input_file[298] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_010_POSCOR_anaTutorial.root";
    input_file[302] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_011_POSCOR_anaTutorial.root";
    input_file[303] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_012_POSCOR_anaTutorial.root"; 
    input_file[307] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_014_POSCOR_anaTutorial.root";
    input_file[308] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_015_POSCOR_anaTutorial.root";
    input_file[312] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_016_POSCOR_anaTutorial.root";
    input_file[313] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_017_POSCOR_anaTutorial.root";
    input_file[317] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_018_POSCOR_anaTutorial.root";
    input_file[318] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_019_POSCOR_anaTutorial.root";
    input_file[322] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_020_POSCOR_anaTutorial.root";
    input_file[323] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_021_POSCOR_anaTutorial.root";
    input_file[326] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_023_POSCOR_anaTutorial.root";
    input_file[329] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_024_POSCOR_anaTutorial.root";
    input_file[332] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_025_POSCOR_anaTutorial.root";
    input_file[335] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_066_POSCOR_anaTutorial.root";
    input_file[338] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_027_POSCOR_anaTutorial.root";
    input_file[341] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_028_POSCOR_anaTutorial.root";
    input_file[344] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_029_POSCOR_anaTutorial.root";
    input_file[347] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_030_POSCOR_anaTutorial.root";
    input_file[350] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_031_POSCOR_anaTutorial.root";
    input_file[353] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_032_POSCOR_anaTutorial.root";
    input_file[356] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_034_POSCOR_anaTutorial.root";
    input_file[359] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_036_POSCOR_anaTutorial.root";
    input_file[362] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_037_POSCOR_anaTutorial.root";
    input_file[365] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_038_POSCOR_anaTutorial.root";
    input_file[368] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_039_POSCOR_anaTutorial.root";
    input_file[371] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_041_POSCOR_anaTutorial.root";
    input_file[374] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_042_POSCOR_anaTutorial.root";
    input_file[377] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_043_POSCOR_anaTutorial.root";
    input_file[380] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_044_POSCOR_anaTutorial.root";
    input_file[383] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_045_POSCOR_anaTutorial.root";
    input_file[386] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_046_POSCOR_anaTutorial.root";
    input_file[389] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_047_POSCOR_anaTutorial.root";
    input_file[392] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_048_POSCOR_anaTutorial.root";
    input_file[395] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_051_POSCOR_anaTutorial.root";
    input_file[398] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_052_POSCOR_anaTutorial.root";
    input_file[401] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_053_POSCOR_anaTutorial.root";
    input_file[404] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_054_POSCOR_anaTutorial.root";
    input_file[407] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_055_POSCOR_anaTutorial.root";
    input_file[410] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_056_POSCOR_anaTutorial.root";
    input_file[413] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_057_POSCOR_anaTutorial.root";
    input_file[416] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_058_POSCOR_anaTutorial.root";
    input_file[419] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_059_POSCOR_anaTutorial.root";
    input_file[422] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_060_POSCOR_anaTutorial.root";
    input_file[425] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_061_POSCOR_anaTutorial.root";
    input_file[428] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_062_POSCOR_anaTutorial.root";
    input_file[431] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_063_POSCOR_anaTutorial.root";
    input_file[434] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_064_POSCOR_anaTutorial.root";
    input_file[437] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K+_eta0-1.1_0-20GeV_100_065_POSCOR_anaTutorial.root";

    input_file[553] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_00_anaTutorial.root";//481+72
    input_file[554] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_00_anaTutorial.root";//481+72+1
    input_file[555] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_00_anaTutorial.root";//481+72+2
    input_file[556] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_01_anaTutorial.root";
    input_file[557] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_01_anaTutorial.root";
    input_file[558] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_01_anaTutorial.root";
    input_file[559] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_02_anaTutorial.root";
    input_file[560] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_02_anaTutorial.root";
    input_file[561] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_02_anaTutorial.root";
    input_file[562] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_03_anaTutorial.root";
    input_file[563] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_03_anaTutorial.root";
    input_file[564] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_03_anaTutorial.root";
    input_file[565] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_04_anaTutorial.root";
    input_file[566] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_04_anaTutorial.root";
    input_file[567] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_04_anaTutorial.root";
    input_file[568] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_05_anaTutorial.root";
    input_file[569] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_05_anaTutorial.root";
    input_file[570] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_05_anaTutorial.root";
    input_file[571] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_06_anaTutorial.root";
    input_file[572] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_06_anaTutorial.root";
    input_file[573] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_06_anaTutorial.root";
    input_file[574] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_07_anaTutorial.root";
    input_file[575] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_07_anaTutorial.root";
    input_file[576] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_07_anaTutorial.root";
    input_file[577] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_08_anaTutorial.root";
    input_file[578] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_08_anaTutorial.root";
    input_file[579] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_08_anaTutorial.root";
    input_file[580] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_09_anaTutorial.root";
    input_file[581] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_09_anaTutorial.root";
    input_file[582] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_09_anaTutorial.root";
    input_file[583] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_010_anaTutorial.root";
    input_file[584] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_010_anaTutorial.root";
    input_file[585] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_010_anaTutorial.root";
    input_file[586] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_011_anaTutorial.root";
    input_file[587] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_011_anaTutorial.root";
    input_file[588] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_011_anaTutorial.root";
    input_file[589] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_012_anaTutorial.root";
    input_file[590] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_012_anaTutorial.root";
    input_file[591] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_012_anaTutorial.root";
    input_file[592] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_013_anaTutorial.root";
    input_file[593] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_013_anaTutorial.root";
    input_file[594] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_013_anaTutorial.root";
    input_file[595] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_014_anaTutorial.root";
    input_file[596] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_014_anaTutorial.root";
    input_file[597] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_014_anaTutorial.root";
    input_file[598] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_015_anaTutorial.root";
    input_file[599] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_015_anaTutorial.root";
    input_file[600] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_015_anaTutorial.root";
    input_file[601] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_016_anaTutorial.root";
    input_file[602] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_016_anaTutorial.root";
    input_file[603] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_016_anaTutorial.root";
    input_file[604] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_017_anaTutorial.root";
    input_file[605] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_017_anaTutorial.root";
    input_file[606] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_017_anaTutorial.root";
    input_file[607] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_018_anaTutorial.root";
    input_file[608] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_018_anaTutorial.root";
    input_file[609] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_018_anaTutorial.root";
    input_file[610] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_019_anaTutorial.root";
    input_file[611] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_019_anaTutorial.root";
    input_file[612] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_019_anaTutorial.root";
    input_file[613] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_020_anaTutorial.root";
    input_file[614] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_020_anaTutorial.root";
    input_file[615] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_020_anaTutorial.root";
    input_file[616] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_021_anaTutorial.root";
    input_file[617] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_021_anaTutorial.root";
    input_file[618] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_021_anaTutorial.root";
    input_file[619] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_022_anaTutorial.root";
    input_file[620] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_022_anaTutorial.root";
    input_file[621] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_022_anaTutorial.root";
    input_file[622] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_023_anaTutorial.root";
    input_file[623] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_023_anaTutorial.root";
    input_file[624] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_023_anaTutorial.root";
    input_file[625] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_024_anaTutorial.root";
    input_file[626] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_024_anaTutorial.root";
    input_file[627] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_024_anaTutorial.root";
    input_file[628] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_025_anaTutorial.root";
    input_file[629] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_025_anaTutorial.root";
    input_file[630] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_025_anaTutorial.root";
    input_file[631] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_026_anaTutorial.root";
    input_file[632] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_026_anaTutorial.root";
    input_file[633] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_026_anaTutorial.root";
    input_file[634] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_027_anaTutorial.root";
    input_file[635] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_027_anaTutorial.root";
    input_file[636] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_027_anaTutorial.root";
    input_file[637] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_028_anaTutorial.root";
    input_file[638] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_028_anaTutorial.root";
    input_file[639] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_028_anaTutorial.root";
    input_file[640] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_029_anaTutorial.root";
    input_file[641] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_029_anaTutorial.root";
    input_file[642] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_029_anaTutorial.root";
    input_file[643] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_030_anaTutorial.root";
    input_file[644] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_030_anaTutorial.root";
    input_file[645] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_030_anaTutorial.root";
    input_file[646] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_031_anaTutorial.root";
    input_file[647] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_031_anaTutorial.root";
    input_file[648] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_031_anaTutorial.root";
    input_file[649] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_032_anaTutorial.root";
    input_file[650] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_032_anaTutorial.root";
    input_file[651] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_032_anaTutorial.root";
    input_file[652] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_033_anaTutorial.root";
    input_file[653] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_033_anaTutorial.root";
    input_file[654] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_033_anaTutorial.root";
    input_file[655] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_034_anaTutorial.root";
    input_file[656] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_034_anaTutorial.root";
    input_file[657] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_034_anaTutorial.root";
    input_file[658] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_035_anaTutorial.root";
    input_file[659] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_035_anaTutorial.root";
    input_file[660] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_035_anaTutorial.root";
    input_file[661] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_036_anaTutorial.root";
    input_file[662] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_036_anaTutorial.root";
    input_file[663] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_036_anaTutorial.root";
    input_file[664] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_037_anaTutorial.root";
    input_file[665] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_037_anaTutorial.root";
    input_file[666] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_037_anaTutorial.root";
    input_file[667] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_038_anaTutorial.root";
    input_file[668] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_038_anaTutorial.root";
    input_file[669] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_038_anaTutorial.root";
    input_file[670] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_039_anaTutorial.root";
    input_file[671] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_039_anaTutorial.root";
    input_file[672] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_039_anaTutorial.root";
    input_file[673] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_040_anaTutorial.root";
    input_file[674] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_040_anaTutorial.root";
    input_file[675] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_040_anaTutorial.root";
    input_file[676] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_041_anaTutorial.root";
    input_file[677] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_041_anaTutorial.root";
    input_file[678] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_041_anaTutorial.root";
    input_file[679] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_042_anaTutorial.root";
    input_file[680] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_042_anaTutorial.root";
    input_file[681] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_042_anaTutorial.root";
    input_file[682] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_043_anaTutorial.root";
    input_file[683] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_043_anaTutorial.root";
    input_file[684] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_043_anaTutorial.root";
    input_file[685] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_044_anaTutorial.root";
    input_file[686] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_044_anaTutorial.root";
    input_file[687] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_044_anaTutorial.root";
    input_file[688] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_045_anaTutorial.root";
    input_file[689] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_045_anaTutorial.root";
    input_file[690] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_045_anaTutorial.root";
    input_file[691] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_046_anaTutorial.root";
    input_file[692] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_046_anaTutorial.root";
    input_file[693] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_046_anaTutorial.root";
    input_file[694] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_047_anaTutorial.root";
    input_file[695] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_047_anaTutorial.root";
    input_file[696] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_047_anaTutorial.root";
    input_file[697] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_048_anaTutorial.root";
    input_file[698] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_048_anaTutorial.root";
    input_file[699] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_048_anaTutorial.root";
    input_file[700] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_049_anaTutorial.root";
    input_file[701] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_049_anaTutorial.root";
    input_file[702] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_049_anaTutorial.root";
    input_file[703] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_050_anaTutorial.root";
    input_file[704] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_050_anaTutorial.root";
    input_file[705] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_050_anaTutorial.root";
    input_file[706] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_051_anaTutorial.root";
    input_file[707] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_051_anaTutorial.root";
    input_file[708] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_051_anaTutorial.root";
    input_file[709] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_052_anaTutorial.root";
    input_file[710] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_052_anaTutorial.root";
    input_file[711] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_052_anaTutorial.root";
    input_file[712] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_053_anaTutorial.root";
    input_file[713] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_053_anaTutorial.root";
    input_file[714] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_053_anaTutorial.root";
    input_file[715] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_054_anaTutorial.root";
    input_file[716] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_054_anaTutorial.root";
    input_file[717] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_054_anaTutorial.root";
    input_file[718] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_055_anaTutorial.root";
    input_file[719] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_055_anaTutorial.root";
    input_file[720] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_055_anaTutorial.root";
    input_file[721] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_056_anaTutorial.root";
    input_file[722] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_056_anaTutorial.root";
    input_file[723] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_056_anaTutorial.root";
    input_file[724] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_057_anaTutorial.root";
    input_file[725] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_057_anaTutorial.root";
    input_file[726] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_057_anaTutorial.root";
    input_file[727] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_058_anaTutorial.root";
    input_file[728] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_058_anaTutorial.root";
    input_file[729] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_058_anaTutorial.root";
    input_file[730] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_059_anaTutorial.root";
    input_file[731] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_059_anaTutorial.root";
    input_file[732] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_059_anaTutorial.root";
    input_file[733] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_060_anaTutorial.root";
    input_file[734] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_060_anaTutorial.root";
    input_file[735] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_060_anaTutorial.root";
    input_file[736] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_061_anaTutorial.root";
    input_file[737] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_061_anaTutorial.root";
    input_file[738] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_061_anaTutorial.root";
    input_file[739] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_062_anaTutorial.root";
    input_file[740] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_062_anaTutorial.root";
    input_file[741] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_062_anaTutorial.root";
    input_file[742] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_063_anaTutorial.root";
    input_file[743] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_063_anaTutorial.root";
    input_file[744] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_063_anaTutorial.root";
    input_file[745] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_064_anaTutorial.root";
    input_file[746] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_064_anaTutorial.root";
    input_file[747] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_064_anaTutorial.root";
    input_file[748] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_065_anaTutorial.root";
    input_file[749] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_065_anaTutorial.root";
    input_file[750] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_065_anaTutorial.root";
    input_file[751] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_pi-_eta0-1.1_2-12GeV_400_066_anaTutorial.root";
    input_file[752] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_2-12GeV_400_066_anaTutorial.root";
    input_file[753] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_066_anaTutorial.root";

    input_file[754] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_067_anaTutorial.root";
    input_file[755] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_068_anaTutorial.root";
    input_file[756] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_069_anaTutorial.root";
    input_file[757] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_070_anaTutorial.root";
    input_file[758] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_071_anaTutorial.root";
    input_file[759] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_072_anaTutorial.root";
    input_file[760] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_073_anaTutorial.root";
    input_file[761] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_074_anaTutorial.root";
    input_file[762] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_075_anaTutorial.root";
    input_file[763] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_076_anaTutorial.root";
    input_file[764] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_077_anaTutorial.root";
    input_file[765] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_078_anaTutorial.root";
    input_file[766] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_079_anaTutorial.root";
    input_file[767] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_080_anaTutorial.root";
    input_file[768] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_081_anaTutorial.root";
    input_file[769] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_082_anaTutorial.root";
    input_file[770] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_083_anaTutorial.root";
    input_file[771] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_084_anaTutorial.root";
    input_file[772] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_085_anaTutorial.root";
    input_file[773] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_086_anaTutorial.root";
    input_file[774] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_087_anaTutorial.root";
    input_file[775] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_088_anaTutorial.root";
    input_file[776] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_089_anaTutorial.root";
    input_file[777] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_090_anaTutorial.root";
    input_file[778] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_091_anaTutorial.root";
    input_file[779] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_092_anaTutorial.root";
    input_file[780] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_093_anaTutorial.root";
    input_file[781] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_094_anaTutorial.root";
    input_file[782] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_095_anaTutorial.root";
    input_file[783] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_096_anaTutorial.root";
    input_file[784] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_097_anaTutorial.root";
    input_file[785] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_098_anaTutorial.root";
    input_file[786] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_099_anaTutorial.root";
    input_file[787] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_0100_anaTutorial.root";
    input_file[788] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_0101_anaTutorial.root";
    input_file[789] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_0102_anaTutorial.root";
    input_file[790] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_0103_anaTutorial.root";
    input_file[791] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_0104_anaTutorial.root";
    input_file[792] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_0105_anaTutorial.root";
    input_file[793] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_2-12GeV_400_0106_anaTutorial.root";

    input_file[794] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_0102_anaTutorial.root";
    input_file[795] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_0103_anaTutorial.root";
    input_file[796] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_0104_anaTutorial.root";
    input_file[797] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_0105_anaTutorial.root";
    input_file[798] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_0106_anaTutorial.root";
    input_file[799] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_072_anaTutorial.root";
    input_file[800] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_073_anaTutorial.root";
    input_file[801] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_074_anaTutorial.root";
    input_file[802] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_075_anaTutorial.root";
    input_file[803] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_076_anaTutorial.root";
    input_file[804] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_077_anaTutorial.root";
    input_file[805] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_078_anaTutorial.root";
    input_file[806] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_079_anaTutorial.root";
    input_file[807] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_080_anaTutorial.root";
    input_file[808] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_081_anaTutorial.root";
    input_file[809] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_082_anaTutorial.root";
    input_file[810] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_083_anaTutorial.root";
    input_file[811] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_084_anaTutorial.root";
    input_file[812] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_085_anaTutorial.root";
    input_file[813] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_086_anaTutorial.root";
    input_file[814] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_087_anaTutorial.root";
    input_file[815] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_088_anaTutorial.root";
    input_file[816] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_089_anaTutorial.root";
    input_file[817] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_090_anaTutorial.root";
    input_file[818] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_091_anaTutorial.root";
    input_file[819] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_092_anaTutorial.root";
    input_file[820] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_093_anaTutorial.root";
    input_file[821] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_094_anaTutorial.root";
    input_file[822] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_095_anaTutorial.root";
    input_file[823] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_096_anaTutorial.root";
    input_file[824] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_097_anaTutorial.root";
    input_file[825] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_098_anaTutorial.root";
    input_file[826] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_099_anaTutorial.root";
    input_file[827] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_0100_anaTutorial.root";
    input_file[828] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_0101_anaTutorial.root";

    input_file[829] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_11-13GeV_400_00_anaTutorial.root";
    input_file[830] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_11-13GeV_400_01_anaTutorial.root";
    input_file[831] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_11-13GeV_400_02_anaTutorial.root";
    input_file[832] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_K-_eta0-1.1_11-13GeV_400_03_anaTutorial.root";
    
    input_file[833] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_9-12GeV_400_00_anaTutorial.root";
    input_file[834] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_9-12GeV_400_01_anaTutorial.root";
    input_file[835] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_9-12GeV_400_02_anaTutorial.root";
    input_file[836] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_9-12GeV_400_03_anaTutorial.root";
    input_file[837] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_9-12GeV_400_04_anaTutorial.root";
    input_file[838] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_9-12GeV_400_05_anaTutorial.root";
    input_file[839] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_9-12GeV_400_06_anaTutorial.root";
    input_file[840] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_9-12GeV_400_07_anaTutorial.root";
    input_file[841] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_9-12GeV_400_08_anaTutorial.root";
    input_file[842] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_9-12GeV_400_09_anaTutorial.root";
    input_file[843] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_9-12GeV_400_010_anaTutorial.root";
    input_file[844] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_9-12GeV_400_011_anaTutorial.root";
    input_file[845] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_antiproton_eta0-1.1_9-12GeV_400_012_anaTutorial.root";
   
  }//single_data   
            

    float EOP=0.01;
    float HOP=0.01;
    float HOM=0.01;
    for(int ifile=189;ifile<209;ifile++){ //Nfile //test:169-188; 189-208
       if(data_single & (ifile==366 or ifile==458 or ifile==450 or ifile==449)) continue;
       
       if(data_single & ifile>=481 && ifile<(481+72)) {
            int ien=ifile-481;
            sprintf(input_file_tem7,"/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_0%d_anaTutorial.root",ien);
            input_file[ifile]=input_file_tem7;
       }
       if(data_embed & ifile>=89 && ifile<=168) continue;

        printf("file name is %s\n",input_file[ifile]);


       /////////////////////////
       TTree *readtree;
       TFile *file4;
       //file4 = new TFile(input_file[ifile]);
        file4 = TFile::Open(input_file[ifile]);
       readtree = (TTree*)file4->Get("tracktree");
        int nmvtx2,nintt2,ntpc2,charge;
        float quality2;
        double trpx,trpy,trpz,trpt,trp,treta,trphi,trdca;
        double cemcdphi,cemcdeta,cemce3x3,cemce5x5,cemce,cemcecore,cemcprob,cemcchi2;
        double hcalindphi,hcalindeta,hcaline3x3,hcaline5x5,hcaline;
        double gflavor2,bimp2;

        int nmvtx,nintt,ntpc,m_charge;
        float quality;
        double m_tr_px,m_tr_py,m_tr_pz,m_tr_pt,m_tr_p,m_tr_eta,m_tr_phi,m_tr_dca;
        double m_cemcdphi,m_cemcdeta,m_cemce3x3,m_cemce5x5,m_cemce,cemc_ecore,cemc_prob,cemc_chi2;
        double m_hcalindphi,m_hcalindeta,m_hcaline3x3,m_hcaline5x5,m_hcaline;
        double truthflavor,bimp;

        readtree->SetBranchAddress( "m_tr_px", &m_tr_px);
        readtree->SetBranchAddress( "m_tr_py", &m_tr_py);
        readtree->SetBranchAddress( "m_tr_pz", &m_tr_pz);
        readtree->SetBranchAddress( "m_tr_pt", &m_tr_pt);
        readtree->SetBranchAddress( "m_tr_p", &m_tr_p);
        readtree->SetBranchAddress( "m_tr_eta", &m_tr_eta);
        readtree->SetBranchAddress( "m_tr_phi", &m_tr_phi);
        readtree->SetBranchAddress( "m_charge", &m_charge);
   
        readtree->SetBranchAddress( "m_cemce3x3", &m_cemce3x3);
        readtree->SetBranchAddress( "m_cemce5x5", &m_cemce5x5);
        readtree->SetBranchAddress( "m_cemce", &m_cemce);
        readtree->SetBranchAddress( "cemc_ecore", &cemc_ecore);
        readtree->SetBranchAddress( "cemc_prob", &cemc_prob);
        readtree->SetBranchAddress( "cemc_chi2", &cemc_chi2);
        readtree->SetBranchAddress( "m_cemcdeta", &m_cemcdeta);
        readtree->SetBranchAddress( "m_cemcdphi", &m_cemcdphi);

        readtree->SetBranchAddress( "m_hcaline3x3", &m_hcaline3x3);
        readtree->SetBranchAddress( "m_hcaline5x5", &m_hcaline5x5);
        readtree->SetBranchAddress( "m_hcaline", &m_hcaline);
        readtree->SetBranchAddress( "m_hcalindeta", &m_hcalindeta);
        readtree->SetBranchAddress( "m_hcalindphi", &m_hcalindphi);

        readtree->SetBranchAddress( "nmvtx", &nmvtx);
        readtree->SetBranchAddress( "nintt", &nintt);
        readtree->SetBranchAddress( "ntpc", &ntpc);
        readtree->SetBranchAddress( "quality", &quality);

        if(data_embed) readtree->SetBranchAddress( "truthflavor", &truthflavor);
        if(data_single & !(ifile<=12 or ifile==113 or ifile==117 or ifile==121 )) readtree->SetBranchAddress( "truthflavor", &truthflavor);
        readtree->SetBranchAddress( "bimp", &bimp);
      

        for (Long64_t ievt=0; ievt<readtree->GetEntries();ievt++) {
         //if (ievt%1000 == 0) cout << "--- ... Processing event: " <<readtree->GetEntries()<<"; "<< ievt << endl;
          //  cout << "--- ... Processing event: " <<readtree->GetEntries()<<"; "<< ievt << endl;
            readtree->GetEntry(ievt);     

            trpx=m_tr_px;
            trpy=m_tr_py;
            trpz=m_tr_pz;
            trpt=m_tr_pt;
            trp=m_tr_p;
            treta=m_tr_eta;
            trphi=m_tr_phi;
            trdca=m_tr_dca;
            charge=m_charge;
   
            cemce3x3=m_cemce3x3;
            cemce5x5=m_cemce5x5;
            cemce=m_cemce;
            cemcecore=cemc_ecore;
            cemcprob=cemc_prob;
            cemcchi2=cemc_chi2;
            cemcdeta=m_cemcdeta;
            cemcdphi=m_cemcdphi;

            hcaline3x3=m_hcaline3x3;
            hcaline5x5=m_hcaline5x5;
            hcaline=m_hcaline;
            hcalindeta=m_hcalindeta;
            hcalindphi=m_hcalindphi;

            nmvtx2=nmvtx;
            nintt2=nintt;
            ntpc2=ntpc;
            quality2=quality; 

            if(data_embed) gflavor2=truthflavor;

            if(data_single & ifile<=12) gflavor2=11;
            if(data_single & !(ifile<=12 or ifile==113 or ifile==117 or ifile==121)) gflavor2=truthflavor;
            if(data_single & (ifile==113 or ifile==117 or ifile==121)) gflavor2=-2122;

            bimp2=bimp;
            cout<<ifile<<"; "<<gflavor2<<"; "<<bimp2<<endl;

            float p2=trp;
            float EOP=cemce3x3/p2;// E3x3/p
            float EcOP=cemcecore/p2;// Ecore/p
            float HOM=hcaline3x3/cemce3x3;// EHcalin/EEmcal
            float dR=TMath::Sqrt(cemcdphi*cemcdphi+cemcdeta*cemcdeta);
            float pt=trpt;
           // std::cout <<ifile<<"; "<<ievt<< "; EOP: " << EOP << " HOM: " <<HOM<< " gflavor2: "<< gflavor2<< std::endl;
           // std::cout <<ifile<<"; "<<ievt<< "; nmvtx2: " << nmvtx2 << " nintt2: " <<nintt2<< " ntpc2: "<< ntpc2<< " quality2: "<< quality2<< " pt: "<< pt<< std::endl;

            h1pt->Fill(pt);
            //h1EOP->Fill(EOP);
            h1EcOP->Fill(EOP);
            h1HOM->Fill(HOM);
            h1CEMCchi2->Fill(cemcchi2);

            if(gflavor2==11) N_raw=N_raw+1;

            if(gflavor2==11 & EOP>0.0 & EOP<20.0 & HOM>0.0 & HOM<20.0 & nmvtx2>0 & nintt2>0 & ntpc2>20 & quality2<10) {
                N_track=N_track+1;
            }

            if(EOP>0.0 & EOP<20.0 & HOM>0.0 & HOM<20.0 & nmvtx2>0 & nintt2>0 & ntpc2>20 & quality2<10 & pt>2.0 & pt<=12.0) {
                h1EOP->Fill(EOP);
            }

         
          if((gflavor2==11 or gflavor2==-2212 or gflavor2==-211 or gflavor2==-321) & nmvtx2>0 && nintt2>0 && quality2<10 & (TMath::Abs(treta)<=1.1) && EOP>0.0 && EOP<20.0 && HOM>0.0 && HOM<20.0 && pt>2.0 && pt<=13.0 && ntpc2>20 && ntpc2<=48 && cemcprob>0.0 && cemcprob<=1.0 && cemcchi2>0.0 && cemcchi2<20.0) {

          //if((gflavor2==11 or gflavor2==-2212 or gflavor2==-211 or gflavor2==-321) & EOP>0.0 & EOP<20.0 & HOM>0.0 & HOM<20.0 & nmvtx2>0 & nintt2>0 & ntpc2>20 & quality2<10 & pt>2.0 & pt<12.0){
           // if((gflavor2==11 or gflavor2==-2212 or gflavor2==-211 or gflavor2==-321) & EcOP>0.0 & EcOP<20.0 & HOM>0.0 & HOM<20.0 & nmvtx2>0 & nintt2>0 & ntpc2>20 & quality2<10 & pt>2.0 & pt<20.0){
           
            if(gflavor2==11) N_track_pt2=N_track_pt2+1;

            if(TMath::Abs(gflavor2)==11) {
                h1EOP_e->Fill(EOP);
                h1HOM_e->Fill(HOM);
                h1CEMCchi2_e->Fill(cemcchi2);
                h1pt_cut->Fill(pt);
            }
            h1flavor_1->Fill(gflavor2);

            // h1EOP->Fill(EOP);
            //h1EcOP->Fill(EOP);
            //h1HOM->Fill(HOM);
            //h1CEMCchi2->Fill(cemcchi2);
           // h1EOP_cut->Fill(EOP);
           // h1pt_cut->Fill(pt);

            var1 = EOP;
            var2 = HOM;
            var3 = cemcchi2;
           // var4 = cemcprob;
           // var5 = ntpc2;
           // var6 = pt;
           // std::cout <<ifile<<"; "<<ievt<< "; var1: " << var1 << " var2: " <<var2 << std::endl;
          //  std::cout <<ifile<<"; "<<ievt<< "; var3: " << var3 << " var3: " <<var3 << std::endl;
          //  std::cout <<ievt<< "; var5: " << var5 << " var6: " <<var6 << std::endl;

           // Return the MVA outputs and fill into histograms

           //  if (Use["CutsGA"]) {
            if (Use["Cuts"]) {//weihu
             // Cuts is a special case: give the desired signal efficienciy
             //   Bool_t passed = reader->EvaluateMVA( "CutsGA method", effS );
                Bool_t passed = reader->EvaluateMVA( "Cuts method", effS );//weihu
                if (passed) nSelCutsGA++;
                histCuts->Fill( reader->EvaluateMVA( "Cuts method", effS ) );//chosed
            }

            if (Use["Likelihood"   ])   histLk     ->Fill( reader->EvaluateMVA( "Likelihood method"    ) );
            if (Use["LikelihoodD"  ])   histLkD    ->Fill( reader->EvaluateMVA( "LikelihoodD method"   ) );
            if (Use["LikelihoodPCA"])   histLkPCA  ->Fill( reader->EvaluateMVA( "LikelihoodPCA method" ) );
            if (Use["LikelihoodKDE"])   histLkKDE  ->Fill( reader->EvaluateMVA( "LikelihoodKDE method" ) );
            if (Use["LikelihoodMIX"])   histLkMIX  ->Fill( reader->EvaluateMVA( "LikelihoodMIX method" ) );
            if (Use["PDERS"        ])   histPD     ->Fill( reader->EvaluateMVA( "PDERS method"         ) );
            if (Use["PDERSD"       ])   histPDD    ->Fill( reader->EvaluateMVA( "PDERSD method"        ) );
            if (Use["PDERSPCA"     ])   histPDPCA  ->Fill( reader->EvaluateMVA( "PDERSPCA method"      ) );
            if (Use["KNN"          ])   histKNN    ->Fill( reader->EvaluateMVA( "KNN method"           ) );
            if (Use["HMatrix"      ])   histHm     ->Fill( reader->EvaluateMVA( "HMatrix method"       ) );
            if (Use["Fisher"       ])   histFi     ->Fill( reader->EvaluateMVA( "Fisher method"        ) );
            if (Use["FisherG"      ])   histFiG    ->Fill( reader->EvaluateMVA( "FisherG method"       ) );
            if (Use["BoostedFisher"])   histFiB    ->Fill( reader->EvaluateMVA( "BoostedFisher method" ) );
            if (Use["LD"           ])   histLD     ->Fill( reader->EvaluateMVA( "LD method"            ) );
            if (Use["MLP"          ])   histNn     ->Fill( reader->EvaluateMVA( "MLP method"           ) );
            if (Use["MLPBFGS"      ])   histNnbfgs ->Fill( reader->EvaluateMVA( "MLPBFGS method"       ) );
            if (Use["MLPBNN"       ])   histNnbnn  ->Fill( reader->EvaluateMVA( "MLPBNN method"        ) );
            if (Use["CFMlpANN"     ])   histNnC    ->Fill( reader->EvaluateMVA( "CFMlpANN method"      ) );
            if (Use["TMlpANN"      ])   histNnT    ->Fill( reader->EvaluateMVA( "TMlpANN method"       ) );
            if (Use["DNN_GPU"      ])   histDnnGpu ->Fill( reader->EvaluateMVA("DNN_GPU method"        ) );
            if (Use["DNN_CPU"      ])   histDnnCpu ->Fill( reader->EvaluateMVA("DNN_CPU method"        ) );
            if (Use["BDT"          ])   histBdt    ->Fill( reader->EvaluateMVA( "BDT method"           ) );
            if (Use["BDTG"         ])   histBdtG   ->Fill( reader->EvaluateMVA( "BDTG method"          ) );
            if (Use["BDTB"         ])   histBdtB   ->Fill( reader->EvaluateMVA( "BDTB method"          ) );
            if (Use["BDTD"         ])   histBdtD   ->Fill( reader->EvaluateMVA( "BDTD method"          ) );
            if (Use["BDTF"         ])   histBdtF   ->Fill( reader->EvaluateMVA( "BDTF method"          ) );
            if (Use["RuleFit"      ])   histRf     ->Fill( reader->EvaluateMVA( "RuleFit method"       ) );
            if (Use["SVM"          ])   histSVM    ->Fill( reader->EvaluateMVA( "SVM method"           ) );
            if (Use["SVM_Gauss"    ])   histSVMG   ->Fill( reader->EvaluateMVA( "SVM_Gauss method"     ) );
            if (Use["SVM_Poly"     ])   histSVMP   ->Fill( reader->EvaluateMVA( "SVM_Poly method"      ) );
            if (Use["SVM_Lin"      ])   histSVML   ->Fill( reader->EvaluateMVA( "SVM_Lin method"       ) );
            if (Use["FDA_MT"       ])   histFDAMT  ->Fill( reader->EvaluateMVA( "FDA_MT method"        ) );
            if (Use["FDA_GA"       ])   histFDAGA  ->Fill( reader->EvaluateMVA( "FDA_GA method"        ) );
            if (Use["Category"     ])   histCat    ->Fill( reader->EvaluateMVA( "Category method"      ) );
            if (Use["Plugin"       ])   histPBdt   ->Fill( reader->EvaluateMVA( "P_BDT method"         ) );

           
         /*
          // Retrieve also per-event error
            if (Use["PDEFoam"]) {
               Double_t val = reader->EvaluateMVA( "PDEFoam method" );
               Double_t err = reader->GetMVAError();
               histPDEFoam   ->Fill( val );
               histPDEFoamErr->Fill( err );
               if (err>1.e-50) histPDEFoamSig->Fill( val/err );
            }

            // Retrieve probability instead of MVA output
            if (Use["Fisher"])   {
               probHistFi  ->Fill( reader->GetProba ( "Fisher method" ) );
               rarityHistFi->Fill( reader->GetRarity( "Fisher method" ) );
            }
         */
            
            if(TMath::Abs(gflavor2)==11) NSall=NSall+1;
            if(TMath::Abs(gflavor2)==11) Nelectron=Nelectron+1;
            if(TMath::Abs(gflavor2)==211) Npion=Npion+1;
            if(TMath::Abs(gflavor2)==2212) Nantiproton=Nantiproton+1;
            if(TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) Nall=Nall+1;

            if(TMath::Abs(gflavor2)==11 & var1>0.912 & var2<0.2) Nelectron_cuts=Nelectron_cuts+1; //traditional cuts: 3 vars：var1>0.908 & var2<0.2; 4 vars：var1>0.909 & var2<0.2; embed var1>0.912 & var2<0.2

            for(int i=0;i<5;i++){
                 Nbimp[i]=4.0*i+2.0;
                 err_Nbimp[i]=2.0;
                 if((TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) & (bimp2>=Nbimp[i]-2.0) & bimp2<(Nbimp[i]+2.0)) Nall_bimp[i]=Nall_bimp[i]+1;
                 //if((TMath::Abs(gflavor2)==211) & (bimp2>=Nbimp[i]-2.0) & bimp2<(Nbimp[i]+2.0)) Nall_bimp[i]=Nall_bimp[i]+1;
                // if((TMath::Abs(gflavor2)==321) & (bimp2>=Nbimp[i]-2.0) & bimp2<(Nbimp[i]+2.0)) Nall_bimp[i]=Nall_bimp[i]+1;
                 //if((TMath::Abs(gflavor2)==2212) & (bimp2>=Nbimp[i]-2.0) & bimp2<(Nbimp[i]+2.0)) Nall_bimp[i]=Nall_bimp[i]+1;
            } 
            for(int i=0;i<5;i++){
                  if(var1>0.912 & var2<0.2){//90%  3 vars：var1>0.908 & var2<0.2; 4 vars：var1>0.909 & var2<0.2;
                      Nbimp[i]=4.0*i+2.0;
                      if((TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) & (bimp>=Nbimp[i]-2.0) & bimp<(Nbimp[i]+2.0)) { //plot for all
                      //if((TMath::Abs(gflavor2)==211) & (bimp>=Nbimp[i]-1.0) & bimp<(Nbimp[i]+1.0)) { //plot for pi-
                     // if((TMath::Abs(gflavor2)==321) & (bimp>=Nbimp[i]-1.0) & bimp<(Nbimp[i]+1.0)) { //plot for K-
                     // if((TMath::Abs(gflavor2)==2212) & (bimp>=Nbimp[i]-1.0) & bimp<(Nbimp[i]+1.0)) { //plot for antiprotpn
                          nall_cuts_bimp[i]=nall_cuts_bimp[i]+1;
                      }

                  } 
              }
            ////////////////////////////////////////////
            for(int i=0;i<10;i++){
                 Npt[i]=2.0*i+2.0;
                 err_Npt[i]=1.0;
                 if((TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0)) Nall_pt[i]=Nall_pt[i]+1;
                 //if((TMath::Abs(gflavor2)==211) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0)) Nall_pt[i]=Nall_pt[i]+1;
                // if((TMath::Abs(gflavor2)==321) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0)) Nall_pt[i]=Nall_pt[i]+1;
                // if((TMath::Abs(gflavor2)==2212) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0)) Nall_pt[i]=Nall_pt[i]+1;
            } 

            /////////////////////////////
            for(int i=0;i<10;i++){
                  if(var1>0.912 & var2<0.2){//90%  3 vars：var1>0.908 & var2<0.2; 4 vars：var1>0.909 & var2<0.2; embed var1>0.912 & var2<0.2
                      Npt[i]=2.0*i+2.0;
                      if((TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0)) { //plot for all
                     // if((TMath::Abs(gflavor2)==211) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0)) { //plot for pi-
                     // if((TMath::Abs(gflavor2)==321) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0)) { //plot for K-
                     // if((TMath::Abs(gflavor2)==2212) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0)) { //plot for antiprotpn
                          nall_cuts_pt[i]=nall_cuts_pt[i]+1;
                      }

                  } 
              }

               for(int i=0;i<10;i++){
                        pt_point[i]=i*2.0+2.0;
                        if(pt>(pt_point[i]-1.0) && pt<(pt_point[i]+1.0) ){
                            if(gflavor2==11) N_electron_pt_cuts[i]=N_electron_pt_cuts[i]+1;
                            if(var1>0.912 & var2<0.2){
                                if(gflavor2==11) NEID_electron_pt_cuts[i]=NEID_electron_pt_cuts[i]+1;               
                            }
                        }
               }
        
            //////////////////////////
            if (Use["LD"]) {
               float select=reader->EvaluateMVA("LD method");
               //std::cout <<"LD select= " << select<< std::endl;
               if(TMath::Abs(gflavor2)==11) h1electron_LD->Fill(select);
               if(TMath::Abs(gflavor2)==11) h1Sall_LD->Fill(select);
               if(TMath::Abs(gflavor2)==211) h1background_pion_LD->Fill(select);
               if(TMath::Abs(gflavor2)==2212) h1background_antiproton_LD->Fill(select);
               if(TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) h1background_all_LD->Fill(select);

               Double_t err = reader->GetMVAError();
               Hist_err_LD->Fill( err );
               //if (err>1.e-50) Hist_Sig_LD->Fill( select/err );
               Hist_Sig_LD->Fill( select/err );
               Hist_prob_LD  ->Fill( reader->GetProba ( "LD method" ) );
               Hist_rarity_LD->Fill( reader->GetRarity( "LD method" ) );
              
              for(int i=0;i<6;i++){
                  if(W_antiproton)Ncut_LD[i]=i*0.1+0.2;//antiproton weight
                  if(W_pion)Ncut_LD[i]=i*0.1+0.15; //pion weight
                  if(W_all & data_embed)Ncut_LD[i]=i*0.085+0.17; //all weight
                  if(W_all & data_single)Ncut_LD[i]=i*0.1+0.17; //all weight
                  if(W_all_ecore & data_single)Ncut_LD[i]=i*0.073+0.2; //all weight
                  if(select>Ncut_LD[i]){
                      // std::cout <<Ncut_LD[i]<< "; LD selected electrons" << std::endl;
                       if(TMath::Abs(gflavor2)==11) nelectron_LD[i]=nelectron_LD[i]+1;
                       if(TMath::Abs(gflavor2)==11) nSall_LD[i]=nSall_LD[i]+1;
                       if(TMath::Abs(gflavor2)==211) npion_LD[i]=npion_LD[i]+1;
                       if(TMath::Abs(gflavor2)==2212) nantiproton_LD[i]=nantiproton_LD[i]+1;
                       if(TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) nall_LD[i]=nall_LD[i]+1;
                      // std::cout << "nelectron_LD= "<<nelectron_LD[i]<< std::endl;
                      // std::cout << "nantiproton_LD= "<<nantiproton_LD[i]<< std::endl;
                   }
                  else{
                      //std::cout << "LD selected background" << std::endl;
                      // if(gflavor2==-211) npion_LD=npion_LD+1;
                      // if(gflavor2==-2212) nantiproton_LD=nantiproton_LD+1;
                   }
               }
                
            }
            /////////////////////////////
            if (Use["BDT"]) {
              float select=reader->EvaluateMVA("BDT method");
              //std::cout <<"BDT select= " << select<< std::endl;
              if(TMath::Abs(gflavor2)==11)h1electron_BDT->Fill(select);
              if(TMath::Abs(gflavor2)==11) h1Sall_BDT->Fill(select);
              if(TMath::Abs(gflavor2)==211)h1background_pion_BDT->Fill(select);
              if(TMath::Abs(gflavor2)==2212)h1background_antiproton_BDT->Fill(select);
              if(TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) h1background_all_BDT->Fill(select);
              if(select>-0.39 & select<-0.35){
                    h1flavor_2->Fill(gflavor2);
                    h1var1_EOP_2->Fill(var1);
                    h1var2_HOM_2->Fill(var2);
                    h1var3_Chi2_2->Fill(var3);
                    h1_p_2->Fill(p2);
                    h1_pt_2->Fill(pt);
                    h1_Eemcal3x3_2->Fill(cemce3x3);
              }
              if(select>-0.49 & select<-0.43){
                   // h1flavor_1->Fill(gflavor2);
                    h1var1_EOP_1->Fill(var1);
                    h1var2_HOM_1->Fill(var2);
                    h1var3_Chi2_1->Fill(var3);
                    h1_p_1->Fill(p2);
                    h1_pt_1->Fill(pt);
                    h1_Eemcal3x3_1->Fill(cemce3x3);
                
              }

               if(TMath::Abs(gflavor2)==11){
                h2_reponse_pt->Fill(select,pt);
                h2_reponse_EOP->Fill(select,EOP);
                h2_reponse_HOM->Fill(select,HOM);
                h2_reponse_chi2->Fill(select,cemcchi2);
               }

              if(TMath::Abs(gflavor2)==11 & select>0.1431) Nelectron_BDT=Nelectron_BDT+1;  //3 vars：select>0.1355; 4vars: select>0.138; embed: select>0.1431

              for(int i=0;i<10;i++){
                  if(select>0.1431){
                      Npt[i]=2.0*i+2.0;
                      if((TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0)) { //plot for all
                     // if((TMath::Abs(gflavor2)==211) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0)) { //plot for pi-
                     // if((TMath::Abs(gflavor2)==321) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0)) { //plot for K-
                     // if((TMath::Abs(gflavor2)==2212) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0)) { //plot for antiprotpn
                          nall_BDT_pt[i]=nall_BDT_pt[i]+1;
                      }

                  } 
              }
            ///////////////////////////////////////////
              for(int i=0;i<5;i++){
                  if(select>0.1431){//90% eID efficency; select>0.1360 for e3x3 cutpt2; select>0.20 for ecore cutpt2; select>0.15 for e3x3; select>0.10 for ecore;  select>0.18 for e3x3 cutpt2 embed
                      Nbimp[i]=4.0*i+2.0;
                      if((TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) & (bimp>=Nbimp[i]-2.0) & bimp<(Nbimp[i]+2.0)) { //plot for all
                     // if((TMath::Abs(gflavor2)==211) & (bimp>=Nbimp[i]-1.0) & bimp<(Nbimp[i]+1.0)) { //plot for pi-
                      //if((TMath::Abs(gflavor2)==321) & (bimp>=Nbimp[i]-1.0) & bimp<(Nbimp[i]+1.0)) { //plot for K-
                      //if((TMath::Abs(gflavor2)==2212) & (bimp>=Nbimp[i]-1.0) & bimp<(Nbimp[i]+1.0)) { //plot for antiprotpn
                          nall_BDT_bimp[i]=nall_BDT_bimp[i]+1;
                      }

                  } 
              }

              for(int i=0;i<10;i++){
                        pt_point[i]=i*2.0+2.0;
                        if(pt>(pt_point[i]-1.0) && pt<(pt_point[i]+1.0) ){
                            if(gflavor2==11) N_electron_pt_BDT[i]=N_electron_pt_BDT[i]+1;
                            if(select>0.1431){
                                if(gflavor2==11) NEID_electron_pt_BDT[i]=NEID_electron_pt_BDT[i]+1;               
                            }
                        }
               }
              
              
              for(int i=0;i<7;i++){
                  if(W_antiproton & data_embed)Ncut_BDT[i]=i*0.1-0.3;//antiproton weight enmbed
                  if(W_antiproton & data_single)Ncut_BDT[i]=i*0.1-0.245;//antiproton weight single
                  if(W_pion)Ncut_BDT[i]=i*0.1-0.2; //pion weight
                  if(W_all & data_embed)Ncut_BDT[i]=i*0.07-0.18; //all weight
                  if(W_all & data_single)Ncut_BDT[i]=i*0.058-0.18; //all weight
                  if(W_all_ecore & data_single)Ncut_BDT[i]=i*0.064-0.20; //all weight most=0.41
                  if(select>Ncut_BDT[i]){
                       //std::cout <<Ncut_BDT[i]<< "; BDT selected electrons" << std::endl;
                       if(TMath::Abs(gflavor2)==11) nelectron_BDT[i]=nelectron_BDT[i]+1;
                       if(TMath::Abs(gflavor2)==11) nSall_BDT[i]=nSall_BDT[i]+1;
                       if(TMath::Abs(gflavor2)==211) npion_BDT[i]=npion_BDT[i]+1;
                       if(TMath::Abs(gflavor2)==2212) nantiproton_BDT[i]=nantiproton_BDT[i]+1;
                       if(TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) nall_BDT[i]=nall_BDT[i]+1;
                      // std::cout << "nelectron_BDT= "<<nelectron_BDT[i]<< std::endl;
                      // std::cout << "nantiproton_BDT= "<<nantiproton_BDT[i]<< std::endl;
                   }
                  else{
                      //std::cout << "BDT selected background" << std::endl;
                      // if(gflavor2==-211) npion_BDT=npion_BDT+1;
                      // if(gflavor2==-2212) nantiproton_BDT=nantiproton_BDT+1;
                   }
               }
                
            }
            ///////////////////////////
            if (Use["SVM"]) {
              float select=reader->EvaluateMVA("SVM method");
              //std::cout <<"SVM select= " << select<< std::endl;
              if(TMath::Abs(gflavor2)==11)h1electron_SVM->Fill(select);
              if(TMath::Abs(gflavor2)==11) h1Sall_SVM->Fill(select);
              if(TMath::Abs(gflavor2)==211)h1background_pion_SVM->Fill(select);
              if(TMath::Abs(gflavor2)==2212)h1background_antiproton_SVM->Fill(select);
              if(TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) h1background_all_SVM->Fill(select);

              if(TMath::Abs(gflavor2)==11 & select>0.7525) Nelectron_SVM=Nelectron_SVM+1; //3 vars：select>0.779; 4vars: select>7578; embed select>0.7525

              for(int i=0;i<10;i++){
                  if(select>0.7525){
                      Npt[i]=2.0*i+2.0;
                      if((TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0)) {
                     // if((TMath::Abs(gflavor2)==211) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0)) { //plot for pi-
                     // if((TMath::Abs(gflavor2)==321) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0)) { //plot for K-
                       // if((TMath::Abs(gflavor2)==2212) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0)) { //plot for antiprotpn
                          nall_SVM_pt[i]=nall_SVM_pt[i]+1;
                      }

                  } 
              }
            //////////////////////////////////////////
              for(int i=0;i<10;i++){
                  if(select>0.7525){//90% eID efficency; select>0.7784 for e3x3 cutpt2; select>0.66 for ecore ; select>0.685 for e3x3; select>0.63 for ecore; select>0.638 for e3x3 cutpt2 embed;
                      Nbimp[i]=4.0*i+2.0;
                      if((TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) & (bimp>=Nbimp[i]-2.0) & bimp<(Nbimp[i]+2.0)) {
                      //if((TMath::Abs(gflavor2)==211) & (bimp>=Nbimp[i]-1.0) & bimp<(Nbimp[i]+1.0)) { //plot for pi-
                      //if((TMath::Abs(gflavor2)==321) & (bimp>=Nbimp[i]-1.0) & bimp<(Nbimp[i]+1.0)) { //plot for K-
                      //  if((TMath::Abs(gflavor2)==2212) & (bimp>=Nbimp[i]-1.0) & bimp<(Nbimp[i]+1.0)) { //plot for antiprotpn
                          nall_SVM_bimp[i]=nall_SVM_bimp[i]+1;
                      }

                  } 
              }

              for(int i=0;i<10;i++){
                        pt_point[i]=i*2.0+2.0;
                        if(pt>(pt_point[i]-1.0) && pt<(pt_point[i]+1.0) ){
                            if(gflavor2==11) N_electron_pt_SVM[i]=N_electron_pt_SVM[i]+1;
                            if(select>0.7525){
                                if(gflavor2==11) NEID_electron_pt_SVM[i]=NEID_electron_pt_SVM[i]+1;               
                            }
                        }
               }


              for(int i=0;i<6;i++){
                  if(W_antiproton)Ncut_SVM[i]=i*0.1+0.3; //antiproton weight
                  if(W_pion)Ncut_SVM[i]=i*0.1+0.32; //pion weight
                  if(W_all & data_embed)Ncut_SVM[i]=i*0.04+0.55; //all weight max=0.8
                  if(W_all & data_single)Ncut_SVM[i]=i*0.055+0.55; //all weight max=0.8
                  if(W_all_ecore & data_single)Ncut_SVM[i]=i*0.03+0.55; //all weight max=0.76
                  if(select>Ncut_SVM[i]){
                    // std::cout << "SVM selected electrons"<< std::endl;
                     if(TMath::Abs(gflavor2)==11) nelectron_SVM[i]=nelectron_SVM[i]+1;
                     if(TMath::Abs(gflavor2)==11) nSall_SVM[i]=nSall_SVM[i]+1;
                     if(TMath::Abs(gflavor2)==211) npion_SVM[i]=npion_SVM[i]+1;
                     if(TMath::Abs(gflavor2)==2212) nantiproton_SVM[i]=nantiproton_SVM[i]+1;
                     if(TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) nall_SVM[i]=nall_SVM[i]+1;
                  }
                  else{
                     //std::cout << "SVM selected background"<< std::endl;
                     //if(gflavor2==-211) npion_SVM=npion_SVM+1;
                     //if(gflavor2==-2212) nantiproton_SVM=nantiproton_SVM+1;
                  }
              }
            }
            /////////////////////////
            if (Use["DNN_CPU"]) {
              float select=reader->EvaluateMVA("DNN_CPU method");
              //std::cout <<"DNN_CPU select= " << select<< std::endl;
              if(TMath::Abs(gflavor2)==11)h1electron_DNN_CPU->Fill(select);
              if(TMath::Abs(gflavor2)==11) h1Sall_DNN_CPU->Fill(select);
              if(TMath::Abs(gflavor2)==211)h1background_pion_DNN_CPU->Fill(select);
              if(TMath::Abs(gflavor2)==2212)h1background_antiproton_DNN_CPU->Fill(select);
              if(TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) h1background_all_DNN_CPU->Fill(select);


              for(int i=0;i<6;i++){
                  if(W_antiproton & data_single)Ncut_DNN_CPU[i]=i*0.07+0.637; //antiproton weight
                  if(W_antiproton & data_embed )Ncut_DNN_CPU[i]=i*0.07+0.633; //antiproton weight
                  if(W_pion)Ncut_DNN_CPU[i]=i*0.07+0.642; //pion weight
                  if(W_all & data_embed)Ncut_DNN_CPU[i]=i*0.065+0.65; //all weight  max=0.985
                  if(W_all & data_single)Ncut_DNN_CPU[i]=i*0.065+0.65; //all weight  max=0.985
                  if(W_all_ecore & data_single)Ncut_DNN_CPU[i]=i*0.066+0.60; //all weight  max=0.985
                  if(select>Ncut_DNN_CPU[i]){
                    // std::cout << "DNN_CPU selected electrons"<< std::endl;
                     if(TMath::Abs(gflavor2)==11) nelectron_DNN_CPU[i]=nelectron_DNN_CPU[i]+1;
                     if(TMath::Abs(gflavor2)==11) nSall_DNN_CPU[i]=nSall_DNN_CPU[i]+1;
                     if(TMath::Abs(gflavor2)==211) npion_DNN_CPU[i]=npion_DNN_CPU[i]+1;
                     if(TMath::Abs(gflavor2)==2212) nantiproton_DNN_CPU[i]=nantiproton_DNN_CPU[i]+1;
                     if(TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) nall_DNN_CPU[i]=nall_DNN_CPU[i]+1;
                  }
                  else{
                     //std::cout << "DNN_CPU selected background"<< std::endl;
                     //if(gflavor2==-211) npion_DNN_CPU=npion_DNN_CPU+1;
                     //if(gflavor2==-2212) nantiproton_DNN_CPU=nantiproton_DNN_CPU+1;
                  }
              }
            }

         }
      }//ievt
      file4->Close();
  }//ifile
    

  float efficiency_electron_LD[10],efficiency_Sall_LD[10],rejection_antiproton_LD[10],rejection_pion_LD[10],rejection_all_LD[10];
  float efficiency_electron_BDT[10],efficiency_Sall_BDT[10],rejection_antiproton_BDT[10],rejection_pion_BDT[10],rejection_all_BDT[10];
  float efficiency_electron_SVM[10],efficiency_Sall_SVM[10],rejection_antiproton_SVM[10],rejection_pion_SVM[10],rejection_all_SVM[10];
  float efficiency_electron_DNN_CPU[10],efficiency_Sall_DNN_CPU[10],rejection_antiproton_DNN_CPU[10],rejection_pion_DNN_CPU[10],rejection_all_DNN_CPU[10];
  float err_efficiency_electron_LD[10],err_efficiency_Sall_LD[10],err_rejection_antiproton_LD[10],err_rejection_pion_LD[10],err_rejection_all_LD[10];
  float err_efficiency_electron_BDT[10],err_efficiency_Sall_BDT[10],err_rejection_antiproton_BDT[10],err_rejection_pion_BDT[10],err_rejection_all_BDT[10];
  float err_efficiency_electron_SVM[10],err_efficiency_Sall_SVM[10],err_rejection_antiproton_SVM[10],err_rejection_pion_SVM[10],err_rejection_all_SVM[10];
  float err_efficiency_electron_DNN_CPU[10],err_efficiency_Sall_DNN_CPU[10],err_rejection_antiproton_DNN_CPU[10],err_rejection_pion_DNN_CPU[10],err_rejection_all_DNN_CPU[10];
  float SBratio_antiproton_LD[10],SBratio_pion_LD[10],SBratio_all_LD[10];
  float SBratio_antiproton_BDT[10],SBratio_pion_BDT[10],SBratio_all_BDT[10];
  float SBratio_antiproton_SVM[10],SBratio_pion_SVM[10],SBratio_all_SVM[10];
  float SBratio_antiproton_DNN_CPU[10],SBratio_pion_DNN_CPU[10],SBratio_all_DNN_CPU[10];
  float rejection_all_SVM_pt[10],err_rejection_all_SVM_pt[10];
  float rejection_all_BDT_pt[10],err_rejection_all_BDT_pt[10];
  float rejection_all_cuts_pt[10],err_rejection_all_cuts_pt[10];
  float rejection_all_SVM_pt_inverse[10],err_rejection_all_SVM_pt_inverse[10];
  float rejection_all_BDT_pt_inverse[10],err_rejection_all_BDT_pt_inverse[10];
  float rejection_all_cuts_pt_inverse[10],err_rejection_all_cuts_pt_inverse[10];

  float rejection_all_SVM_bimp[10],err_rejection_all_SVM_bimp[10];
  float rejection_all_BDT_bimp[10],err_rejection_all_BDT_bimp[10];
  float rejection_all_cuts_bimp[10],err_rejection_all_cuts_bimp[10];
  float rejection_all_SVM_bimp_inverse[10],err_rejection_all_SVM_bimp_inverse[10];
  float rejection_all_BDT_bimp_inverse[10],err_rejection_all_BDT_bimp_inverse[10];
  float rejection_all_cuts_bimp_inverse[10],err_rejection_all_cuts_bimp_inverse[10];

  float aa_pt_N[10],err_aa_pt_N[10];
  float cc_pt_N_cuts[10],err_cc_pt_N_cuts[10], cc_pt_N_BDT[10],err_cc_pt_N_BDT[10], cc_pt_N_SVM[10],err_cc_pt_N_SVM[10];

  for(int i=0;i<10;i++){
     efficiency_electron_LD[i]=0.0;
     efficiency_Sall_LD[i]=0.0;
     rejection_antiproton_LD[i]=0.0;
     rejection_pion_LD[i]=0.0;
     rejection_all_LD[i]=0.0;
     efficiency_electron_BDT[i]=0.0;
     efficiency_Sall_BDT[i]=0.0;
     rejection_antiproton_BDT[i]=0.0;
     rejection_pion_BDT[i]=0.0;
     rejection_all_BDT[i]=0.0;
     efficiency_electron_SVM[i]=0.0;
     efficiency_Sall_SVM[i]=0.0;
     rejection_antiproton_SVM[i]=0.0;
     rejection_pion_SVM[i]=0.0;
     rejection_all_SVM[i]=0.0;
     efficiency_electron_DNN_CPU[i]=0.0;
     efficiency_Sall_DNN_CPU[i]=0.0;
     rejection_antiproton_DNN_CPU[i]=0.0;
     rejection_pion_DNN_CPU[i]=0.0;
     rejection_all_DNN_CPU[i]=0.0;

     err_efficiency_electron_LD[i]=0.0;
     err_efficiency_Sall_LD[i]=0.0;
     err_rejection_antiproton_LD[i]=0.0;
     err_rejection_pion_LD[i]=0.0;
     err_rejection_all_LD[i]=0.0;
     err_efficiency_electron_BDT[i]=0.0;
     err_efficiency_Sall_BDT[i]=0.0;
     err_rejection_antiproton_BDT[i]=0.0;
     err_rejection_pion_BDT[i]=0.0;
     err_rejection_all_BDT[i]=0.0;
     err_efficiency_electron_SVM[i]=0.0;
     err_efficiency_Sall_SVM[i]=0.0;
     err_rejection_antiproton_SVM[i]=0.0;
     err_rejection_pion_SVM[i]=0.0;
     err_rejection_all_SVM[i]=0.0;
     err_efficiency_electron_DNN_CPU[i]=0.0;
     err_efficiency_Sall_DNN_CPU[i]=0.0;
     err_rejection_antiproton_DNN_CPU[i]=0.0;
     err_rejection_pion_DNN_CPU[i]=0.0;
     err_rejection_all_DNN_CPU[i]=0.0;

     SBratio_antiproton_LD[i]=0.0;
     SBratio_antiproton_BDT[i]=0.0;
     SBratio_antiproton_SVM[i]=0.0;
     SBratio_antiproton_DNN_CPU[i]=0.0;
     SBratio_pion_LD[i]=0.0;
     SBratio_pion_BDT[i]=0.0;
     SBratio_pion_SVM[i]=0.0;
     SBratio_pion_DNN_CPU[i]=0.0;
     SBratio_all_LD[i]=0.0;
     SBratio_all_BDT[i]=0.0;
     SBratio_all_SVM[i]=0.0;
     SBratio_all_DNN_CPU[i]=0.0;

     rejection_all_SVM_pt[i]=0.0;
     err_rejection_all_SVM_pt[i]=0.0;

     rejection_all_BDT_pt[i]=0.0;
     err_rejection_all_BDT_pt[i]=0.0;

     rejection_all_cuts_pt[i]=0.0;
     err_rejection_all_cuts_pt[i]=0.0;

     rejection_all_SVM_pt_inverse[i]=0.0;
     err_rejection_all_SVM_pt_inverse[i]=0.0;

     rejection_all_BDT_pt_inverse[i]=0.0;
     err_rejection_all_BDT_pt_inverse[i]=0.0;

     rejection_all_cuts_pt_inverse[i]=0.0;
     err_rejection_all_cuts_pt_inverse[i]=0.0;
    ///////////////////
     rejection_all_SVM_bimp[i]=0.0;
     err_rejection_all_SVM_bimp[i]=0.0;

     rejection_all_BDT_bimp[i]=0.0;
     err_rejection_all_BDT_bimp[i]=0.0;

     rejection_all_cuts_bimp[i]=0.0;
     err_rejection_all_cuts_bimp[i]=0.0;

     rejection_all_SVM_bimp_inverse[i]=0.0;
     err_rejection_all_SVM_bimp_inverse[i]=0.0;

     rejection_all_BDT_bimp_inverse[i]=0.0;
     err_rejection_all_BDT_bimp_inverse[i]=0.0;

     rejection_all_cuts_bimp_inverse[i]=0.0;
     err_rejection_all_cuts_bimp_inverse[i]=0.0;

     aa_pt_N[i]=0.0;
     err_aa_pt_N[i]=0.0;
     cc_pt_N_cuts[i]=0.0;
     err_cc_pt_N_cuts[i]=0.0;
     cc_pt_N_BDT[i]=0.0;
     err_cc_pt_N_BDT[i]=0.0;
     cc_pt_N_SVM[i]=0.0;
     err_cc_pt_N_SVM[i]=0.0;
    
  }

   for(int i=0;i<10;i++){
    aa_pt_N[i]=pt_point[i];
	err_aa_pt_N[i]=1.0;
    if(N_electron_pt_cuts[i]>0 && NEID_electron_pt_cuts[i]>0){
        cc_pt_N_cuts[i]=1.0*NEID_electron_pt_cuts[i]/N_electron_pt_cuts[i];
        err_cc_pt_N_cuts[i]=1.0*TMath::Sqrt((1.0/NEID_electron_pt_cuts[i]+1.0/N_electron_pt_cuts[i]))*cc_pt_N_cuts[i];
    }
    if(N_electron_pt_BDT[i]>0 && NEID_electron_pt_BDT[i]>0){
        cc_pt_N_BDT[i]=1.0*NEID_electron_pt_BDT[i]/N_electron_pt_BDT[i];
        err_cc_pt_N_BDT[i]=1.0*TMath::Sqrt((1.0/NEID_electron_pt_BDT[i]+1.0/N_electron_pt_BDT[i]))*cc_pt_N_BDT[i];
    }
    if(N_electron_pt_SVM[i]>0 && NEID_electron_pt_SVM[i]>0){
        cc_pt_N_SVM[i]=1.0*NEID_electron_pt_SVM[i]/N_electron_pt_SVM[i];
        err_cc_pt_N_SVM[i]=1.0*TMath::Sqrt((1.0/NEID_electron_pt_SVM[i]+1.0/N_electron_pt_SVM[i]))*cc_pt_N_SVM[i];
    }
  }

  for(int i=0;i<6;i++){
     if(Nelectron>0 & nelectron_LD[i]>0){
         efficiency_electron_LD[i]=1.0*nelectron_LD[i]/Nelectron;
         err_efficiency_electron_LD[i]=1.0*TMath::Sqrt((1.0/nelectron_LD[i]+1.0/Nelectron))*efficiency_electron_LD[i];
     }
     if(NSall>0 & nSall_LD[i]>0){
         efficiency_Sall_LD[i]=1.0*nSall_LD[i]/NSall;
         err_efficiency_Sall_LD[i]=1.0*TMath::Sqrt((1.0/nSall_LD[i]+1.0/NSall))*efficiency_Sall_LD[i];
     }
     if(Nantiproton>0 & nantiproton_LD[i]>0){
         rejection_antiproton_LD[i]=1.0*Nantiproton/nantiproton_LD[i];
         err_rejection_antiproton_LD[i]=1.0*TMath::Sqrt((1.0/Nantiproton+1.0/nantiproton_LD[i]))*rejection_antiproton_LD[i];
         SBratio_antiproton_LD[i]=1.0*nelectron_LD[i]/TMath::Sqrt(nantiproton_LD[i]+nelectron_LD[i]);
         //SBratio_antiproton_LD[i]=1.0*nelectron_LD[i]/(nantiproton_LD[i]+nelectron_LD[i]);
     }
     if(Npion>0 & npion_LD[i]>0){
         rejection_pion_LD[i]=1.0*Npion/npion_LD[i];
         err_rejection_pion_LD[i]=1.0*TMath::Sqrt((1.0/Npion+1.0/npion_LD[i]))*rejection_pion_LD[i];
         SBratio_pion_LD[i]=1.0*nelectron_LD[i]/TMath::Sqrt(npion_LD[i]+nelectron_LD[i]);
         //SBratio_pion_LD[i]=1.0*nelectron_LD[i]/(npion_LD[i]+nelectron_LD[i]);
     }
     if(Nall>0 & nall_LD[i]>0){
         rejection_all_LD[i]=1.0*Nall/nall_LD[i];
         err_rejection_all_LD[i]=1.0*TMath::Sqrt((1.0/Nall+1.0/nall_LD[i]))*rejection_all_LD[i];
         SBratio_all_LD[i]=1.0*nSall_LD[i]/TMath::Sqrt(nall_LD[i]+nSall_LD[i]);
         //SBratio_all_LD[i]=1.0*nSall_LD[i]/(nall_LD[i]+nSall_LD[i]);
     }
  }

  for(int i=0;i<7;i++){
     if(Nelectron>0 & nelectron_BDT[i]>0){
         efficiency_electron_BDT[i]=1.0*nelectron_BDT[i]/Nelectron;
         err_efficiency_electron_BDT[i]=1.0*TMath::Sqrt((1.0/nelectron_BDT[i]+1.0/Nelectron))*efficiency_electron_BDT[i];
     }
     if(NSall>0 & nSall_BDT[i]>0){
         efficiency_Sall_BDT[i]=1.0*nSall_BDT[i]/NSall;
         err_efficiency_Sall_BDT[i]=1.0*TMath::Sqrt((1.0/nSall_BDT[i]+1.0/NSall))*efficiency_Sall_BDT[i];
     }
     if(Nantiproton>0 & nantiproton_BDT[i]>0){
         rejection_antiproton_BDT[i]=1.0*Nantiproton/nantiproton_BDT[i];
         err_rejection_antiproton_BDT[i]=1.0*TMath::Sqrt((1.0/Nantiproton+1.0/nantiproton_BDT[i]))*rejection_antiproton_BDT[i];
         SBratio_antiproton_BDT[i]=1.0*nelectron_BDT[i]/TMath::Sqrt(nantiproton_BDT[i]+nelectron_BDT[i]);
         //SBratio_antiproton_BDT[i]=1.0*nelectron_BDT[i]/(nantiproton_BDT[i]+nelectron_BDT[i]);
     }
     if(Npion>0 & npion_BDT[i]>0){
         rejection_pion_BDT[i]=1.0*Npion/npion_BDT[i];
         err_rejection_pion_BDT[i]=1.0*TMath::Sqrt((1.0/Npion+1.0/npion_BDT[i]))*rejection_pion_BDT[i];
         SBratio_pion_BDT[i]=1.0*nelectron_BDT[i]/TMath::Sqrt(npion_BDT[i]+nelectron_BDT[i]);
         //SBratio_pion_BDT[i]=1.0*nelectron_BDT[i]/(npion_BDT[i]+nelectron_BDT[i]);
     }
     if(Nall>0 & nall_BDT[i]>0){
         rejection_all_BDT[i]=1.0*Nall/nall_BDT[i];
         err_rejection_all_BDT[i]=1.0*TMath::Sqrt((1.0/Nall+1.0/nall_BDT[i]))*rejection_all_BDT[i];
         SBratio_all_BDT[i]=1.0*nSall_BDT[i]/TMath::Sqrt(nall_BDT[i]+nSall_BDT[i]);
         //SBratio_all_BDT[i]=1.0*nSall_BDT[i]/(nall_BDT[i]+nSall_BDT[i]);
     }
  }
  for(int i=0;i<9;i++){
     if(Nall_pt[i]>0 & nall_BDT_pt[i]>0){
         rejection_all_BDT_pt[i]=1.0*Nall_pt[i]/nall_BDT_pt[i];
         err_rejection_all_BDT_pt[i]=1.0*TMath::Sqrt((1.0/Nall_pt[i]+1.0/nall_BDT_pt[i]))*rejection_all_BDT_pt[i];

         rejection_all_BDT_pt_inverse[i]=1.0*nall_BDT_pt[i]/Nall_pt[i];
         err_rejection_all_BDT_pt_inverse[i]=1.0*TMath::Sqrt((1.0/Nall_pt[i]+1.0/nall_BDT_pt[i]))*rejection_all_BDT_pt_inverse[i];
     }
  }
  for(int i=0;i<5;i++){
     if(Nall_bimp[i]>0 & nall_BDT_bimp[i]>0){
         rejection_all_BDT_bimp[i]=1.0*Nall_bimp[i]/nall_BDT_bimp[i];
         err_rejection_all_BDT_bimp[i]=1.0*TMath::Sqrt((1.0/Nall_bimp[i]+1.0/nall_BDT_bimp[i]))*rejection_all_BDT_bimp[i];

         rejection_all_BDT_bimp_inverse[i]=1.0*nall_BDT_bimp[i]/Nall_bimp[i];
         err_rejection_all_BDT_bimp_inverse[i]=1.0*TMath::Sqrt((1.0/Nall_bimp[i]+1.0/nall_BDT_bimp[i]))*rejection_all_BDT_bimp_inverse[i];
     }
  }

  for(int i=0;i<9;i++){
     if(Nall_pt[i]>0 & nall_cuts_pt[i]>0){
         rejection_all_cuts_pt[i]=1.0*Nall_pt[i]/nall_cuts_pt[i];
         err_rejection_all_cuts_pt[i]=1.0*TMath::Sqrt((1.0/Nall_pt[i]+1.0/nall_cuts_pt[i]))*rejection_all_cuts_pt[i];

         rejection_all_cuts_pt_inverse[i]=1.0*nall_cuts_pt[i]/Nall_pt[i];
         err_rejection_all_cuts_pt_inverse[i]=1.0*TMath::Sqrt((1.0/Nall_pt[i]+1.0/nall_cuts_pt[i]))*rejection_all_cuts_pt_inverse[i];
     }
  }
  for(int i=0;i<5;i++){
     if(Nall_bimp[i]>0 & nall_cuts_bimp[i]>0){
         rejection_all_cuts_bimp[i]=1.0*Nall_bimp[i]/nall_cuts_bimp[i];
         err_rejection_all_cuts_bimp[i]=1.0*TMath::Sqrt((1.0/Nall_bimp[i]+1.0/nall_cuts_bimp[i]))*rejection_all_cuts_bimp[i];

         rejection_all_cuts_bimp_inverse[i]=1.0*nall_cuts_bimp[i]/Nall_bimp[i];
         err_rejection_all_cuts_bimp_inverse[i]=1.0*TMath::Sqrt((1.0/Nall_bimp[i]+1.0/nall_cuts_bimp[i]))*rejection_all_cuts_bimp_inverse[i];
     }
  }

  for(int i=0;i<6;i++){
     if(Nelectron>0 & nelectron_SVM[i]>0){
         efficiency_electron_SVM[i]=1.0*nelectron_SVM[i]/Nelectron;
         err_efficiency_electron_SVM[i]=1.0*TMath::Sqrt((1.0/nelectron_SVM[i]+1.0/Nelectron))*efficiency_electron_SVM[i];
     }
     if(NSall>0 & nSall_SVM[i]>0){
         efficiency_Sall_SVM[i]=1.0*nSall_SVM[i]/NSall;
         err_efficiency_Sall_SVM[i]=1.0*TMath::Sqrt((1.0/nSall_SVM[i]+1.0/NSall))*efficiency_Sall_SVM[i];
     }
     if(Nantiproton>0 & nantiproton_SVM[i]>0){
         rejection_antiproton_SVM[i]=1.0*Nantiproton/nantiproton_SVM[i];
         err_rejection_antiproton_SVM[i]=1.0*TMath::Sqrt((1.0/Nantiproton+1.0/nantiproton_SVM[i]))*rejection_antiproton_SVM[i];
         SBratio_antiproton_SVM[i]=1.0*nelectron_SVM[i]/TMath::Sqrt(nantiproton_SVM[i]+nelectron_SVM[i]);
         //SBratio_antiproton_SVM[i]=1.0*nelectron_SVM[i]/(nantiproton_SVM[i]+nelectron_SVM[i]);
     }
     if(Npion>0 & npion_SVM[i]>0){
         rejection_pion_SVM[i]=1.0*Npion/npion_SVM[i];
         err_rejection_pion_SVM[i]=1.0*TMath::Sqrt((1.0/Npion+1.0/npion_SVM[i]))*rejection_pion_SVM[i];
         SBratio_pion_SVM[i]=1.0*nelectron_SVM[i]/TMath::Sqrt(npion_SVM[i]+nelectron_SVM[i]);
         //SBratio_pion_SVM[i]=1.0*nelectron_SVM[i]/(npion_SVM[i]+nelectron_SVM[i]);
     }
     if(Nall>0 & nall_SVM[i]>0){
         rejection_all_SVM[i]=1.0*Nall/nall_SVM[i];
         err_rejection_all_SVM[i]=1.0*TMath::Sqrt((1.0/Nall+1.0/nall_SVM[i]))*rejection_all_SVM[i];
         SBratio_all_SVM[i]=1.0*nSall_SVM[i]/TMath::Sqrt(nall_SVM[i]+nSall_SVM[i]);
         //SBratio_all_SVM[i]=1.0*nSall_SVM[i]/(nall_SVM[i]+nSall_SVM[i]);
     }
  }
  for(int i=0;i<9;i++){
     if(Nall_pt[i]>0 & nall_SVM_pt[i]>0){
         rejection_all_SVM_pt[i]=1.0*Nall_pt[i]/nall_SVM_pt[i];
         err_rejection_all_SVM_pt[i]=1.0*TMath::Sqrt((1.0/Nall_pt[i]+1.0/nall_SVM_pt[i]))*rejection_all_SVM_pt[i];

         rejection_all_SVM_pt_inverse[i]=1.0*nall_SVM_pt[i]/Nall_pt[i];
         err_rejection_all_SVM_pt_inverse[i]=1.0*TMath::Sqrt((1.0/Nall_pt[i]+1.0/nall_SVM_pt[i]))*rejection_all_SVM_pt_inverse[i];
     }
  }
  for(int i=0;i<5;i++){
     if(Nall_bimp[i]>0 & nall_SVM_bimp[i]>0){
         rejection_all_SVM_bimp[i]=1.0*Nall_bimp[i]/nall_SVM_bimp[i];
         err_rejection_all_SVM_bimp[i]=1.0*TMath::Sqrt((1.0/Nall_bimp[i]+1.0/nall_SVM_bimp[i]))*rejection_all_SVM_bimp[i];

         rejection_all_SVM_bimp_inverse[i]=1.0*nall_SVM_bimp[i]/Nall_bimp[i];
         err_rejection_all_SVM_bimp_inverse[i]=1.0*TMath::Sqrt((1.0/Nall_bimp[i]+1.0/nall_SVM_bimp[i]))*rejection_all_SVM_bimp_inverse[i];
     }
  }

  for(int i=0;i<6;i++){
     if(Nelectron>0 & nelectron_DNN_CPU[i]>0){
         efficiency_electron_DNN_CPU[i]=1.0*nelectron_DNN_CPU[i]/Nelectron;
         err_efficiency_electron_DNN_CPU[i]=1.0*TMath::Sqrt((1.0/nelectron_DNN_CPU[i]+1.0/Nelectron))*efficiency_electron_DNN_CPU[i];
     }
     if(NSall>0 & nSall_DNN_CPU[i]>0){
         efficiency_Sall_DNN_CPU[i]=1.0*nSall_DNN_CPU[i]/NSall;
         err_efficiency_Sall_DNN_CPU[i]=1.0*TMath::Sqrt((1.0/nSall_DNN_CPU[i]+1.0/NSall))*efficiency_Sall_DNN_CPU[i];
     }
     if(Nantiproton>0 & nantiproton_DNN_CPU[i]>0){
         rejection_antiproton_DNN_CPU[i]=1.0*Nantiproton/nantiproton_DNN_CPU[i];
         err_rejection_antiproton_DNN_CPU[i]=1.0*TMath::Sqrt((1.0/Nantiproton+1.0/nantiproton_DNN_CPU[i]))*rejection_antiproton_DNN_CPU[i];
         SBratio_antiproton_DNN_CPU[i]=1.0*nelectron_DNN_CPU[i]/TMath::Sqrt(nantiproton_DNN_CPU[i]+nelectron_DNN_CPU[i]);
         //SBratio_antiproton_DNN_CPU[i]=1.0*nelectron_DNN_CPU[i]/(nantiproton_DNN_CPU[i]+nelectron_DNN_CPU[i]);
     }
     if(Npion>0 & npion_DNN_CPU[i]>0){
         rejection_pion_DNN_CPU[i]=1.0*Npion/npion_DNN_CPU[i];
         err_rejection_pion_DNN_CPU[i]=1.0*TMath::Sqrt((1.0/Npion+1.0/npion_DNN_CPU[i]))*rejection_pion_DNN_CPU[i];
         SBratio_pion_DNN_CPU[i]=1.0*nelectron_DNN_CPU[i]/TMath::Sqrt(npion_DNN_CPU[i]+nelectron_DNN_CPU[i]);
         //SBratio_pion_DNN_CPU[i]=1.0*nelectron_DNN_CPU[i]/(npion_DNN_CPU[i]+nelectron_DNN_CPU[i]);
     }
     if(Nall>0 & nall_DNN_CPU[i]>0){
         rejection_all_DNN_CPU[i]=1.0*Nall/nall_DNN_CPU[i];
         err_rejection_all_DNN_CPU[i]=1.0*TMath::Sqrt((1.0/Nall+1.0/nall_DNN_CPU[i]))*rejection_all_DNN_CPU[i];
         SBratio_all_DNN_CPU[i]=1.0*nSall_DNN_CPU[i]/TMath::Sqrt(nall_DNN_CPU[i]+nSall_DNN_CPU[i]);
         //SBratio_all_DNN_CPU[i]=1.0*nSall_DNN_CPU[i]/(nall_DNN_CPU[i]+nSall_DNN_CPU[i]);
     }
  }

/*
  for(int i=0;i<7;i++){
     if(Nelectron>0 & nantiproton_BDT[i]>0){
         efficiency_electron_BDT[i]=1.0*nelectron_BDT[i]/Nelectron;
         err_efficiency_electron_BDT[i]=1.0*TMath::Sqrt((1.0/nelectron_BDT[i]+1.0/Nelectron))*efficiency_electron_BDT[i];
         rejection_antiproton_BDT[i]=1.0*Nantiproton/nantiproton_BDT[i];
         err_rejection_antiproton_BDT[i]=1.0*TMath::Sqrt((1.0/Nantiproton+1.0/nantiproton_BDT[i]))*rejection_antiproton_BDT[i];
         //SBratio_antiproton_BDT[i]=1.0*nelectron_BDT[i]/TMath::Sqrt(nantiproton_BDT[i]+nelectron_BDT[i]);
         SBratio_antiproton_BDT[i]=1.0*nelectron_BDT[i]/(nantiproton_BDT[i]+nelectron_BDT[i]);
     }
     if(Npion>0 & npion_BDT[i]>0){
         rejection_pion_BDT[i]=1.0*Npion/npion_BDT[i];
         err_rejection_pion_BDT[i]=1.0*TMath::Sqrt((1.0/Npion+1.0/npion_BDT[i]))*rejection_pion_BDT[i];
         //SBratio_pion_BDT[i]=1.0*nelectron_BDT[i]/TMath::Sqrt(npion_BDT[i]+nelectron_BDT[i]);
         SBratio_pion_BDT[i]=1.0*nelectron_BDT[i]/(npion_BDT[i]+nelectron_BDT[i]);
     }
  }

  for(int i=0;i<6;i++){
     if(Nelectron>0 & nantiproton_SVM[i]>0){
         efficiency_electron_SVM[i]=1.0*nelectron_SVM[i]/Nelectron;
         err_efficiency_electron_SVM[i]=1.0*TMath::Sqrt((1.0/nelectron_SVM[i]+1.0/Nelectron))*efficiency_electron_SVM[i];
         rejection_antiproton_SVM[i]=1.0*Nantiproton/nantiproton_SVM[i];
         err_rejection_antiproton_SVM[i]=1.0*TMath::Sqrt((1.0/Nantiproton+1.0/nantiproton_SVM[i]))*rejection_antiproton_SVM[i];
         //SBratio_antiproton_SVM[i]=1.0*nelectron_SVM[i]/TMath::Sqrt(nantiproton_SVM[i]+nelectron_SVM[i]);
         SBratio_antiproton_SVM[i]=1.0*nelectron_SVM[i]/(nantiproton_SVM[i]+nelectron_SVM[i]);
     }
     if(Npion>0 & npion_SVM[i]>0){
         rejection_pion_SVM[i]=1.0*Npion/npion_SVM[i];
         err_rejection_pion_SVM[i]=1.0*TMath::Sqrt((1.0/Npion+1.0/npion_SVM[i]))*rejection_pion_SVM[i];
         //SBratio_pion_SVM[i]=1.0*nelectron_SVM[i]/TMath::Sqrt(npion_SVM[i]+nelectron_SVM[i]);
         SBratio_pion_SVM[i]=1.0*nelectron_SVM[i]/(npion_SVM[i]+nelectron_SVM[i]);
     }
  }

  for(int i=0;i<6;i++){
     if(Nelectron>0 & nantiproton_DNN_CPU[i]>0){
         efficiency_electron_DNN_CPU[i]=1.0*nelectron_DNN_CPU[i]/Nelectron;
         err_efficiency_electron_DNN_CPU[i]=1.0*TMath::Sqrt((1.0/nelectron_DNN_CPU[i]+1.0/Nelectron))*efficiency_electron_DNN_CPU[i];
         rejection_antiproton_DNN_CPU[i]=1.0*Nantiproton/nantiproton_DNN_CPU[i];
         err_rejection_antiproton_DNN_CPU[i]=1.0*TMath::Sqrt((1.0/Nantiproton+1.0/nantiproton_DNN_CPU[i]))*rejection_antiproton_DNN_CPU[i];
         //SBratio_antiproton_DNN_CPU[i]=1.0*nelectron_DNN_CPU[i]/TMath::Sqrt(nantiproton_DNN_CPU[i]+nelectron_DNN_CPU[i]);
         SBratio_antiproton_DNN_CPU[i]=1.0*nelectron_DNN_CPU[i]/(nantiproton_DNN_CPU[i]+nelectron_DNN_CPU[i]);
     }
     if(Npion>0 & npion_DNN_CPU[i]>0){
         rejection_pion_DNN_CPU[i]=1.0*Npion/npion_DNN_CPU[i];
         err_rejection_pion_DNN_CPU[i]=1.0*TMath::Sqrt((1.0/Npion+1.0/npion_DNN_CPU[i]))*rejection_pion_DNN_CPU[i];
         //SBratio_pion_DNN_CPU[i]=1.0*nelectron_DNN_CPU[i]/TMath::Sqrt(npion_DNN_CPU[i]+nelectron_DNN_CPU[i]);
         SBratio_pion_DNN_CPU[i]=1.0*nelectron_DNN_CPU[i]/(npion_DNN_CPU[i]+nelectron_DNN_CPU[i]);
     }
  }
  */
  /*
  std::cout << "LD electron efficiency=: "<<1.0*nelectron_LD[4]/Nelectron<< std::endl;
  std::cout << "BDT electron efficiency=: "<<1.0*nelectron_BDT[4]/Nelectron<< std::endl;
  std::cout << "SVM electron efficiency=: "<<1.0*nelectron_SVM[4]/Nelectron<< std::endl;
  std::cout << "DNN_CPU electron efficiency=: "<<1.0*nelectron_DNN_CPU[4]/Nelectron<< std::endl;

  std::cout << "LD Sall efficiency=: "<<1.0*nSall_LD[4]/NSall<< std::endl;
  std::cout << "BDT Sall efficiency=: "<<1.0*nSall_BDT[4]/NSall<< std::endl;
  std::cout << "SVM Sall efficiency=: "<<1.0*nSall_SVM[4]/NSall<< std::endl;
  std::cout << "DNN_CPU Sall efficiency=: "<<1.0*nSall_DNN_CPU[4]/NSall<< std::endl;

  std::cout << "LD pion rejection=: "<<SBratio_pion_LD[4]<< std::endl;
  std::cout << "BDT pion rejection=: "<<SBratio_pion_BDT[4]<< std::endl;
  std::cout << "SVM pion rejection=: "<<SBratio_pion_SVM[4]<< std::endl;
  std::cout << "DNN_CPU pion rejection=: "<<SBratio_pion_DNN_CPU[4]<< std::endl;

  std::cout << "LD antiproton rejection=: "<<SBratio_antiproton_LD[4]<< std::endl;
  std::cout << "BDT antiproton rejection=: "<<SBratio_antiproton_BDT[4]<< std::endl;
  std::cout << "SVM antiproton rejection=: "<<SBratio_antiproton_SVM[4]<< std::endl;
  std::cout << "DNN_CPU antiproton rejection=: "<<SBratio_antiproton_DNN_CPU[4]<< std::endl;

  std::cout << "LD all rejection=: "<<SBratio_all_LD[4]<< std::endl;
  std::cout << "BDT all rejection=: "<<SBratio_all_BDT[4]<< std::endl;
  std::cout << "SVM all rejection=: "<<SBratio_all_SVM[4]<< std::endl;
  std::cout << "DNN_CPU all rejection=: "<<SBratio_all_DNN_CPU[4]<< std::endl;
*/

   // Get elapsed time
   sw.Stop();
   std::cout << "--- End of event loop: "; sw.Print();
/*
   // Get efficiency for cuts classifier
   if (Use["CutsGA"]) std::cout << "--- Efficiency for CutsGA method: " << double(nSelCutsGA)/ntp_track->GetEntries()
                                << " (for a required signal efficiency of " << effS << ")" << std::endl;

   if (Use["CutsGA"]) {

      // test: retrieve cuts for particular signal efficiency
      // CINT ignores dynamic_casts so we have to use a cuts-secific Reader function to acces the pointer
      TMVA::MethodCuts* mcuts = reader->FindCutsMVA( "CutsGA method" ) ;

      if (mcuts) {
         std::vector<Double_t> cutsMin;
         std::vector<Double_t> cutsMax;
         mcuts->GetCuts( 0.8, cutsMin, cutsMax );
         std::cout << "--- -------------------------------------------------------------" << std::endl;
         std::cout << "--- Retrieve cut values for signal efficiency of 0.8 from Reader" << std::endl;
         for (UInt_t ivar=0; ivar<cutsMin.size(); ivar++) {
            std::cout << "... Cut: "
                      << cutsMin[ivar]
                      << " < \""
                      << mcuts->GetInputVar(ivar)
                      << "\" <= "
                      << cutsMax[ivar] << std::endl;
         }
         std::cout << "--- -------------------------------------------------------------" << std::endl;
      }
   }
*/
   // Write histograms
   std::cout << "Cuts Selected signal number: "<< nSelCutsGA << std::endl;

   TFile *target;

   if(data_embed){
       if(W_antiproton) target  = new TFile( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_embed_cutpt2_12_N/TMVApp_antiproton_weight_embed.root","RECREATE" );
       if(W_pion) target  = new TFile( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_embed_cutpt2_12_N/TMVApp_pion_weight_embed.root","RECREATE" );
       if(W_Kion) target  = new TFile( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_embed_cutpt2_12_N/TMVApp_Kion_weight_embed.root","RECREATE" );
       if(W_all) target  = new TFile( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_embed_cutpt2_12_N/TMVApp_all_weight_embed.root","RECREATE" );
   }
   if(data_single){
       if(W_antiproton) target  = new TFile( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_cutpt2_12_N/TMVApp_antiproton_weight_single.root","RECREATE" );
       if(W_pion) target  = new TFile( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_cutpt2_12_N/TMVApp_pion_weight_single.root","RECREATE" );
       if(W_Kion) target  = new TFile( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_cutpt2_12_N/TMVApp_Kion_weight_single.root","RECREATE" );
       if(W_all) target  = new TFile( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_cutpt2_12_N/TMVApp_all_weight_single.root","RECREATE" );
       if(W_all_ecore) target  = new TFile( "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_cutpt2_12_N/TMVApp_all_weight_single_ecore.root","RECREATE" );
   }
   if (Use["Cuts"   ])         histCuts   ->Write();//weihu
   if (Use["Likelihood"   ])   histLk     ->Write();
   if (Use["LikelihoodD"  ])   histLkD    ->Write();
   if (Use["LikelihoodPCA"])   histLkPCA  ->Write();
   if (Use["LikelihoodKDE"])   histLkKDE  ->Write();
   if (Use["LikelihoodMIX"])   histLkMIX  ->Write();
   if (Use["PDERS"        ])   histPD     ->Write();
   if (Use["PDERSD"       ])   histPDD    ->Write();
   if (Use["PDERSPCA"     ])   histPDPCA  ->Write();
   if (Use["KNN"          ])   histKNN    ->Write();
   if (Use["HMatrix"      ])   histHm     ->Write();
   if (Use["Fisher"       ])   histFi     ->Write();
   if (Use["FisherG"      ])   histFiG    ->Write();
   if (Use["BoostedFisher"])   histFiB    ->Write();
   if (Use["LD"           ])   histLD     ->Write();
   if (Use["MLP"          ])   histNn     ->Write();
   if (Use["MLPBFGS"      ])   histNnbfgs ->Write();
   if (Use["MLPBNN"       ])   histNnbnn  ->Write();
   if (Use["CFMlpANN"     ])   histNnC    ->Write();
   if (Use["TMlpANN"      ])   histNnT    ->Write();
   if (Use["DNN_GPU"      ])   histDnnGpu ->Write();
   if (Use["DNN_CPU"      ])   histDnnCpu ->Write();
   if (Use["BDT"          ])   histBdt    ->Write();
   if (Use["BDTG"         ])   histBdtG   ->Write();
   if (Use["BDTB"         ])   histBdtB   ->Write();
   if (Use["BDTD"         ])   histBdtD   ->Write();
   if (Use["BDTF"         ])   histBdtF   ->Write();
   if (Use["RuleFit"      ])   histRf     ->Write();
   if (Use["SVM"          ])   histSVM    ->Write();
   if (Use["SVM_Gauss"    ])   histSVMG   ->Write();
   if (Use["SVM_Poly"     ])   histSVMP   ->Write();
   if (Use["SVM_Lin"      ])   histSVML   ->Write();
   if (Use["FDA_MT"       ])   histFDAMT  ->Write();
   if (Use["FDA_GA"       ])   histFDAGA  ->Write();
   if (Use["Category"     ])   histCat    ->Write();
   if (Use["Plugin"       ])   histPBdt   ->Write();

   h1electron_LD->Write();
   h1Sall_LD->Write();
   h1background_LD->Write();
   h1background_pion_LD->Write();
   h1background_antiproton_LD->Write();
   h1background_all_LD->Write();
   Hist_err_LD->Write();
   Hist_prob_LD->Write();
   Hist_rarity_LD->Write();
   Hist_Sig_LD->Write();

   h1electron_BDT->Write();
   h1Sall_BDT->Write();
   h1background_BDT->Write();
   h1background_pion_BDT->Write();
   h1background_antiproton_BDT->Write();
   h1background_all_BDT->Write();

   h1electron_SVM->Write();
   h1Sall_SVM->Write();
   h1background_SVM->Write();
   h1background_pion_SVM->Write();
   h1background_antiproton_SVM->Write();
   h1background_all_SVM->Write();

   h1electron_DNN_CPU->Write();
   h1Sall_DNN_CPU->Write();
   h1background_DNN_CPU->Write();
   h1background_pion_DNN_CPU->Write();
   h1background_antiproton_DNN_CPU->Write();
   h1background_all_DNN_CPU->Write();

   h1EOP->Write();
   h1EOP_cut->Write();
   h1EcOP->Write();
   h1HOM->Write();
   h1CEMCchi2->Write();

   h1EOP_e->Write();
   h1HOM_e->Write();
   h1CEMCchi2_e->Write();

   h1pt->Write();
   h1pt_cut->Write();

   h1flavor_1->Write();
   h1flavor_2->Write();
   h1var1_EOP_1->Write();
   h1var2_HOM_1->Write();
   h1var3_Chi2_1->Write();
   h1var1_EOP_2->Write();
   h1var2_HOM_2->Write();
   h1var3_Chi2_2->Write();

   h1_p_1->Write();
   h1_pt_1->Write();
   h1_Eemcal3x3_1->Write();
   h1_p_2->Write();
   h1_pt_2->Write();
   h1_Eemcal3x3_2->Write();

   h2_reponse_pt->Write();
   h2_reponse_EOP->Write();
   h2_reponse_HOM->Write();
   h2_reponse_chi2->Write();


   // Write also error and significance histos
   if (Use["PDEFoam"]) { histPDEFoam->Write(); histPDEFoamErr->Write(); histPDEFoamSig->Write(); }

   // Write also probability hists
   if (Use["Fisher"]) { if (probHistFi != 0) probHistFi->Write(); if (rarityHistFi != 0) rarityHistFi->Write(); }
   target->Close();

   std::cout << "--- Created root file: \"TMVApp.root\" containing the MVA output histograms" << std::endl;

   delete reader;

   std::cout << "==> TMVAClassificationApplication is done!" << std::endl;


   //////////////////////////////////plot1
   gROOT->LoadMacro("sPhenixStyle.C");
   SetsPhenixStyle();

   TCanvas *canv= new TCanvas("canv","Cali Canvas",2700,1800);
   canv->Divide(3,2);
   canv->cd(1);
   TPad *pad1 = new TPad("pad1","pad1",0,0,0.99,0.99);
   pad1->Draw();    
   pad1->cd();
   //pad1->SetFrameLineWidth(2);
   //pad1->SetFrameLineColor(1);
   Float_t Xmin,Xmax;
   Float_t Ymin,Ymax;
   Ymin=1.0;
   if(data_embed)Ymax=1000.0;//embed
   if(data_single)Ymax=10000.0;//single
   Xmin=0.7;
   Xmax=1.1; 
   gPad->SetLogy();

   TH1F *hframe0;
   hframe0=gPad->DrawFrame(Xmin,Ymin,Xmax,Ymax);
   hframe0->GetXaxis()->SetTitle("eID_efficiency"); 
   // hframe0->GetXaxis()->CenterTitle();	
   hframe0->GetYaxis()->SetTitle("Antiproton Rejection");
   // hframe0->GetYaxis()->CenterTitle();
   TGraphErrors *gP1,*gP2,*gP3,*gDNN1;
   gP1=new TGraphErrors(7,efficiency_electron_BDT,rejection_antiproton_BDT,err_efficiency_electron_BDT,err_rejection_antiproton_BDT);
   gP1->SetMarkerStyle(26);
   gP1->SetMarkerColor(2);
   gP1->SetLineColor(2);
   gP1->SetLineStyle(1);
   gP1->SetLineWidth(1.2);
   gP1->SetMarkerSize(3.7);
   gP1->Draw("pl");

   gP2=new TGraphErrors(6,efficiency_electron_SVM,rejection_antiproton_SVM,err_efficiency_electron_SVM,err_rejection_antiproton_SVM);
   gP2->SetMarkerStyle(26);
   gP2->SetMarkerColor(3);
   gP2->SetLineColor(3);
   gP2->SetLineStyle(1);
   gP2->SetLineWidth(1.2);
   gP2->SetMarkerSize(3.7);
   gP2->Draw("pl");

   gP3=new TGraphErrors(6,efficiency_electron_LD,rejection_antiproton_LD,err_efficiency_electron_LD,err_rejection_antiproton_LD);
   gP3->SetMarkerStyle(26);
   gP3->SetMarkerColor(4);
   gP3->SetLineColor(4);
   gP3->SetLineStyle(1);
   gP3->SetLineWidth(1.2);
   gP3->SetMarkerSize(3.7);
   gP3->Draw("pl");

   gDNN1=new TGraphErrors(6,efficiency_electron_DNN_CPU,rejection_antiproton_DNN_CPU,err_efficiency_electron_DNN_CPU,err_rejection_antiproton_DNN_CPU);
   gDNN1->SetMarkerStyle(26);
   gDNN1->SetMarkerColor(6);
   gDNN1->SetLineColor(6);
   gDNN1->SetLineStyle(1);
   gDNN1->SetLineWidth(1.2);
   gDNN1->SetMarkerSize(3.7);
   gDNN1->Draw("pl");

   TLegend *legP1 =new TLegend(0.30,0.25,0.50,0.50);
   legP1->AddEntry(gP1,"  BDT","lep");
   legP1->AddEntry(gP2,"  SVM","lep");
   legP1->AddEntry(gP3,"  LD","lep");
   legP1->AddEntry(gDNN1,"  DNN","lep");
   legP1->Draw();

   TLegend *legtitle1 =new TLegend(0.20,0.86,0.63,0.90,"#it{#bf{sPHENIX}} Simulation");
   legtitle1->SetTextSize(0.05);
   legtitle1->Draw();
  
   TLegend *legtitle11;
   if(data_embed){
       if(W_antiproton) legtitle11 =new TLegend(0.20,0.84,0.30,0.87," MVA_antiproton_weights/ Embed");
       if(W_pion) legtitle11 =new TLegend(0.20,0.84,0.30,0.87," MVA_pion_weights/ Embed");
       if(W_Kion) legtitle11 =new TLegend(0.20,0.84,0.30,0.87," MVA_Kion_weights/ Embed");
       if(W_all) legtitle11 =new TLegend(0.20,0.84,0.30,0.87,"  Embed");
   }
   if(data_single){
       if(W_antiproton) legtitle11 =new TLegend(0.20,0.81,0.39,0.84," MVA_antiproton_weights/ Single particle");
       if(W_pion) legtitle11 =new TLegend(0.20,0.81,0.39,0.84," MVA_pion_weights/ Single particle");
       if(W_Kion) legtitle11 =new TLegend(0.20,0.81,0.39,0.84," MVA_Kion_weights/ Single particle");
       if(W_all) legtitle11 =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
       if(W_all_ecore) legtitle11 =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
   }
   legtitle11->SetTextSize(0.04);
   legtitle11->Draw();

//////////////////////////////////////////////////
   canv->cd(2);
   TPad *pad2 = new TPad("pad2","pad2",0,0,0.99,0.99);
   pad2->Draw();    
   pad2->cd();
   Ymin=1.0;
   if(data_embed)Ymax=1000.0;//embed
   if(data_single)Ymax=10000.0;//single
   Xmin=0.7;
   Xmax=1.1;  
   gPad->SetLogy();
   hframe0=gPad->DrawFrame(Xmin,Ymin,Xmax,Ymax);
   hframe0->GetXaxis()->SetTitle("eID_efficiency"); 
   // hframe0->GetXaxis()->CenterTitle();	
   hframe0->GetYaxis()->SetTitle("pion- Rejection");
   // hframe0->GetYaxis()->CenterTitle();
   TGraphErrors *gP4,*gP5,*gP6,*gDNN2;
   gP4=new TGraphErrors(7,efficiency_electron_BDT,rejection_pion_BDT,err_efficiency_electron_BDT,err_rejection_pion_BDT);
   gP4->SetMarkerStyle(26);
   gP4->SetMarkerColor(2);
   gP4->SetLineColor(2);
   gP4->SetLineStyle(1);
   gP4->SetLineWidth(1.2);
   gP4->SetMarkerSize(3.7);
   gP4->Draw("pl");

   gP5=new TGraphErrors(6,efficiency_electron_SVM,rejection_pion_SVM,err_efficiency_electron_SVM,err_rejection_pion_SVM);
   gP5->SetMarkerStyle(26);
   gP5->SetMarkerColor(3);
   gP5->SetLineColor(3);
   gP5->SetLineStyle(1);
   gP5->SetLineWidth(1.2);
   gP5->SetMarkerSize(3.7);
   gP5->Draw("pl");

   gP6=new TGraphErrors(6,efficiency_electron_LD,rejection_pion_LD,err_efficiency_electron_LD,err_rejection_pion_LD);
   gP6->SetMarkerStyle(26);
   gP6->SetMarkerColor(4);
   gP6->SetLineColor(4);
   gP6->SetLineStyle(1);
   gP6->SetLineWidth(1.2);
   gP6->SetMarkerSize(3.7);
   gP6->Draw("pl");

   gDNN2=new TGraphErrors(6,efficiency_electron_DNN_CPU,rejection_pion_DNN_CPU,err_efficiency_electron_DNN_CPU,err_rejection_pion_DNN_CPU);
   gDNN2->SetMarkerStyle(26);
   gDNN2->SetMarkerColor(6);
   gDNN2->SetLineColor(6);
   gDNN2->SetLineStyle(1);
   gDNN2->SetLineWidth(1.2);
   gDNN2->SetMarkerSize(3.7);
   gDNN2->Draw("pl");

   TLegend *legP2 =new TLegend(0.3,0.25,0.5,0.5);
   legP2->AddEntry(gP4,"  BDT","lep");
   legP2->AddEntry(gP5,"  SVM","lep");
   legP2->AddEntry(gP6,"  LD","lep");
   legP2->AddEntry(gDNN2,"  DNN","lep");
   legP2->Draw();

   TLegend *legtitle2 =new TLegend(0.20,0.86,0.63,0.90,"#it{#bf{sPHENIX}} Simulation");
   legtitle2->SetTextSize(0.05);
   legtitle2->Draw();
  
   TLegend *legtitle21;
   if(data_embed){
       if(W_antiproton) legtitle21 =new TLegend(0.20,0.84,0.30,0.87," MVA_antiproton_weights/ Embed");
       if(W_pion) legtitle21 =new TLegend(0.20,0.84,0.30,0.87," MVA_pion_weights/ Embed");
       if(W_Kion) legtitle21 =new TLegend(0.20,0.84,0.30,0.87," MVA_Kion_weights/ Embed");
       if(W_all) legtitle21 =new TLegend(0.20,0.84,0.30,0.87,"  Embed");
   }
   if(data_single){
       if(W_antiproton) legtitle21 =new TLegend(0.20,0.81,0.39,0.84," MVA_antiproton_weights/ Single particle");
       if(W_pion) legtitle21 =new TLegend(0.20,0.81,0.39,0.84," MVA_pion_weights/ Single particle");
       if(W_Kion) legtitle21 =new TLegend(0.20,0.81,0.39,0.84," MVA_Kion_weights/ Single particle");
       if(W_all) legtitle21 =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
       if(W_all_ecore) legtitle21 =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
   }
   legtitle21->SetTextSize(0.04);
   legtitle21->Draw();
//////////////////////////////////////////////////
   canv->cd(3);
   TPad *padall = new TPad("padall","padall",0,0,0.99,0.99);
   padall->Draw();    
   padall->cd();
   Ymin=1.0;
   if(data_embed)Ymax=1000.0;//embed
   if(data_single)Ymax=10000.0;//single
   Xmin=0.7;
   Xmax=1.1;  
   gPad->SetLogy();
   hframe0=gPad->DrawFrame(Xmin,Ymin,Xmax,Ymax);
   hframe0->GetXaxis()->SetTitle("eID_efficiency"); 
   // hframe0->GetXaxis()->CenterTitle();	
   hframe0->GetYaxis()->SetTitle("Hadron(-) Rejection");
   // hframe0->GetYaxis()->CenterTitle();
   TGraphErrors *gPall_BDT,*gPall_SVM,*gPall_LD,*gPall_DNN;
   gPall_BDT=new TGraphErrors(7,efficiency_Sall_BDT,rejection_all_BDT,err_efficiency_Sall_BDT,err_rejection_all_BDT);
   gPall_BDT->SetMarkerStyle(26);
   gPall_BDT->SetMarkerColor(2);
   gPall_BDT->SetLineColor(2);
   gPall_BDT->SetLineStyle(1);
   gPall_BDT->SetLineWidth(1.2);
   gPall_BDT->SetMarkerSize(3.7);
   gPall_BDT->Draw("pl");

   gPall_SVM=new TGraphErrors(6,efficiency_Sall_SVM,rejection_all_SVM,err_efficiency_Sall_SVM,err_rejection_all_SVM);
   gPall_SVM->SetMarkerStyle(26);
   gPall_SVM->SetMarkerColor(3);
   gPall_SVM->SetLineColor(3);
   gPall_SVM->SetLineStyle(1);
   gPall_SVM->SetLineWidth(1.2);
   gPall_SVM->SetMarkerSize(3.7);
   gPall_SVM->Draw("pl");

   gPall_LD=new TGraphErrors(6,efficiency_Sall_LD,rejection_all_LD,err_efficiency_Sall_LD,err_rejection_all_LD);
   gPall_LD->SetMarkerStyle(26);
   gPall_LD->SetMarkerColor(4);
   gPall_LD->SetLineColor(4);
   gPall_LD->SetLineStyle(1);
   gPall_LD->SetLineWidth(1.2);
   gPall_LD->SetMarkerSize(3.7);
   gPall_LD->Draw("pl");

   gPall_DNN=new TGraphErrors(6,efficiency_Sall_DNN_CPU,rejection_all_DNN_CPU,err_efficiency_Sall_DNN_CPU,err_rejection_all_DNN_CPU);
   gPall_DNN->SetMarkerStyle(26);
   gPall_DNN->SetMarkerColor(6);
   gPall_DNN->SetLineColor(6);
   gPall_DNN->SetLineStyle(1);
   gPall_DNN->SetLineWidth(1.2);
   gPall_DNN->SetMarkerSize(3.7);
   gPall_DNN->Draw("pl");

   TLegend *legPall =new TLegend(0.3,0.25,0.5,0.5);
   legPall->AddEntry(gPall_BDT,"  BDT","lep");
   legPall->AddEntry(gPall_SVM,"  SVM","lep");
   legPall->AddEntry(gPall_LD,"  LD","lep");
   legPall->AddEntry(gPall_DNN,"  DNN","lep");
   legPall->Draw();

   TLegend *legtitleall0 =new TLegend(0.20,0.86,0.63,0.90,"#it{#bf{sPHENIX}} Simulation");
   legtitleall0->SetTextSize(0.05);
   legtitleall0->Draw();
  
   TLegend *legtitleall;
   if(data_embed){
       if(W_antiproton) legtitleall =new TLegend(0.20,0.84,0.30,0.87," MVA_antiproton_weights/ Embed");
       if(W_pion) legtitleall =new TLegend(0.20,0.84,0.30,0.87," MVA_pion_weights/ Embed");
       if(W_Kion) legtitleall =new TLegend(0.20,0.84,0.30,0.87," MVA_Kion_weights/ Embed");
       if(W_all) legtitleall =new TLegend(0.20,0.84,0.30,0.87,"  Embed");
   }
   if(data_single){
       if(W_antiproton) legtitleall =new TLegend(0.20,0.81,0.39,0.84," MVA_antiproton_weights/ Single particle");
       if(W_pion) legtitleall =new TLegend(0.20,0.81,0.39,0.84," MVA_pion_weights/ Single particle");
       if(W_Kion) legtitleall =new TLegend(0.20,0.81,0.39,0.84," MVA_Kion_weights/ Single particle");
       if(W_all) legtitleall =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
       if(W_all_ecore) legtitleall =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
   }
   legtitleall->SetTextSize(0.04);
   legtitleall->Draw();

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
   canv->cd(4);
   TPad *pad3 = new TPad("pad3","pad3",0,0,0.99,0.99);
   pad3->Draw();    
   pad3->cd();
   
   Xmin=0.7;
   Xmax=1.1;  

   if(data_single) Ymin=100.0;//single
   if(data_embed) Ymin=20.0;//embed
   if(data_single) Ymax=180.0;//single
   if(data_embed) Ymax=60.0;//embed 
   /*
   if(data_single) Ymin=0.7;//single
   if(data_embed) Ymin=0.7;//embed
   if(data_single) Ymax=1.1;//single
   if(data_embed) Ymax=1.1;//embed 
   */
  // gPad->SetLogy();
   hframe0=gPad->DrawFrame(Xmin,Ymin,Xmax,Ymax);
   hframe0->GetXaxis()->SetTitle("eID_efficiency"); 
   // hframe0->GetXaxis()->CenterTitle();	
  // hframe0->GetYaxis()->SetTitle("Antiproton S/(S+B)");
   hframe0->GetYaxis()->SetTitle("Antiproton S/Sqrt(S+B)");
   // hframe0->GetYaxis()->CenterTitle();
   TGraph *gP11,*gP21,*gP31,*gDNN3;
   gP11=new TGraph(7,efficiency_electron_BDT,SBratio_antiproton_BDT);
   gP11->SetMarkerStyle(26);
   gP11->SetMarkerColor(2);
   gP11->SetLineColor(2);
   gP11->SetLineStyle(1);
   gP11->SetLineWidth(1.2);
   gP11->SetMarkerSize(3.7);
   gP11->Draw("pl");

   gP21=new TGraph(6,efficiency_electron_SVM,SBratio_antiproton_SVM);
   gP21->SetMarkerStyle(26);
   gP21->SetMarkerColor(3);
   gP21->SetLineColor(3);
   gP21->SetLineStyle(1);
   gP21->SetLineWidth(1.2);
   gP21->SetMarkerSize(3.7);
   gP21->Draw("pl");

   gP31=new TGraphErrors(6,efficiency_electron_LD,SBratio_antiproton_LD);
   gP31->SetMarkerStyle(26);
   gP31->SetMarkerColor(4);
   gP31->SetLineColor(4);
   gP31->SetLineStyle(1);
   gP31->SetLineWidth(1.2);
   gP31->SetMarkerSize(3.7);
   gP31->Draw("pl");

   gDNN3=new TGraph(6,efficiency_electron_DNN_CPU,SBratio_antiproton_DNN_CPU);
   gDNN3->SetMarkerStyle(26);
   gDNN3->SetMarkerColor(6);
   gDNN3->SetLineColor(6);
   gDNN3->SetLineStyle(1);
   gDNN3->SetLineWidth(1.2);
   gDNN3->SetMarkerSize(3.7);
   gDNN3->Draw("pl");

  // TLegend *legP3 =new TLegend(0.3,0.6,0.5,0.8);
   TLegend *legP3 =new TLegend(0.30,0.25,0.50,0.50);
   legP3->AddEntry(gP11,"  BDT","lep");
   legP3->AddEntry(gP21,"  SVM","lep");
   legP3->AddEntry(gP31,"  LD","lep");
   legP3->AddEntry(gDNN3,"  DNN","lep");
   legP3->Draw();

   TLegend *legtitle3 =new TLegend(0.20,0.86,0.63,0.90,"#it{#bf{sPHENIX}} Simulation");
   legtitle3->SetTextSize(0.05);
   legtitle3->Draw();
  
   TLegend *legtitle31;
   if(data_embed){
       if(W_antiproton) legtitle31 =new TLegend(0.20,0.84,0.30,0.87," MVA_antiproton_weights/ Embed");
       if(W_pion) legtitle31 =new TLegend(0.20,0.84,0.30,0.87," MVA_pion_weights/ Embed");
       if(W_Kion) legtitle31 =new TLegend(0.20,0.84,0.30,0.87," MVA_Kion_weights/ Embed");
       if(W_all) legtitle31 =new TLegend(0.20,0.84,0.30,0.87,"  Embed");
   }
   if(data_single){
       if(W_antiproton) legtitle31 =new TLegend(0.20,0.81,0.39,0.84," MVA_antiproton_weights/ Single particle");
       if(W_pion) legtitle31 =new TLegend(0.20,0.81,0.39,0.84," MVA_pion_weights/ Single particle");
       if(W_Kion) legtitle31 =new TLegend(0.20,0.81,0.39,0.84," MVA_Kion_weights/ Single particle");
       if(W_all) legtitle31 =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
       if(W_all_ecore) legtitle31 =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
   }
   legtitle31->SetTextSize(0.04);
   legtitle31->Draw();

//////////////////////////////////////////////////
   canv->cd(5);
   TPad *pad4 = new TPad("pad4","pad4",0,0,0.99,0.99);
   pad4->Draw();    
   pad4->cd();
   
   
   if(data_single) Ymin=100.0;//single
   if(data_embed) Ymin=20.0;//embed
   if(data_single) Ymax=180.0;//single
   if(data_embed) Ymax=60.0;//embed 
   /*
   if(data_single) Ymin=0.7;//single
   if(data_embed) Ymin=0.7;//embed
   if(data_single) Ymax=1.1;//single
   if(data_embed) Ymax=1.1;//embed  
   */
   Xmin=0.7;
   Xmax=1.1;  
  // gPad->SetLogy();
   hframe0=gPad->DrawFrame(Xmin,Ymin,Xmax,Ymax);
   hframe0->GetXaxis()->SetTitle("eID_efficiency"); 
   // hframe0->GetXaxis()->CenterTitle();	
  // hframe0->GetYaxis()->SetTitle("pion S/(S+B)");
   hframe0->GetYaxis()->SetTitle("pion- S/Sqrt(S+B)");
   // hframe0->GetYaxis()->CenterTitle();
   TGraph *gP41,*gP51,*gP61,*gDNN4;
   gP41=new TGraph(7,efficiency_electron_BDT,SBratio_pion_BDT);
   gP41->SetMarkerStyle(26);
   gP41->SetMarkerColor(2);
   gP41->SetLineColor(2);
   gP41->SetLineStyle(1);
   gP41->SetLineWidth(1.2);
   gP41->SetMarkerSize(3.7);
   gP41->Draw("pl");

   gP51=new TGraph(6,efficiency_electron_SVM,SBratio_pion_SVM);
   gP51->SetMarkerStyle(26);
   gP51->SetMarkerColor(3);
   gP51->SetLineColor(3);
   gP51->SetLineStyle(1);
   gP51->SetLineWidth(1.2);
   gP51->SetMarkerSize(3.7);
   gP51->Draw("pl");

   gP61=new TGraphErrors(6,efficiency_electron_LD,SBratio_pion_LD);
   gP61->SetMarkerStyle(26);
   gP61->SetMarkerColor(4);
   gP61->SetLineColor(4);
   gP61->SetLineStyle(1);
   gP61->SetLineWidth(1.2);
   gP61->SetMarkerSize(3.7);
   gP61->Draw("pl");

   gDNN4=new TGraph(6,efficiency_electron_DNN_CPU,SBratio_pion_DNN_CPU);
   gDNN4->SetMarkerStyle(26);
   gDNN4->SetMarkerColor(6);
   gDNN4->SetLineColor(6);
   gDNN4->SetLineStyle(1);
   gDNN4->SetLineWidth(1.2);
   gDNN4->SetMarkerSize(3.7);
   gDNN4->Draw("pl");

   // TLegend *legP4 =new TLegend(0.3,0.6,0.5,0.8);
   TLegend *legP4 =new TLegend(0.30,0.25,0.50,0.50);
   legP4->AddEntry(gP41,"  BDT","lep");
   legP4->AddEntry(gP51,"  SVM","lep");
   legP4->AddEntry(gP61,"  LD","lep");
   legP4->AddEntry(gDNN4,"  DNN","lep");
   legP4->Draw();

   TLegend *legtitle4 =new TLegend(0.20,0.86,0.63,0.90,"#it{#bf{sPHENIX}} Simulation");
   legtitle4->SetTextSize(0.05);
   legtitle4->Draw();
  
   TLegend *legtitle41;
    if(data_embed){
       if(W_antiproton) legtitle41 =new TLegend(0.20,0.84,0.30,0.87," MVA_antiproton_weights/ Embed");
       if(W_pion) legtitle41 =new TLegend(0.20,0.84,0.30,0.87," MVA_pion_weights/ Embed");
       if(W_Kion) legtitle41 =new TLegend(0.20,0.84,0.30,0.87," MVA_Kion_weights/ Embed");
       if(W_all) legtitle41 =new TLegend(0.20,0.84,0.30,0.87,"  Embed");
   }
   if(data_single){
       if(W_antiproton) legtitle41 =new TLegend(0.20,0.81,0.39,0.84," MVA_antiproton_weights/ Single particle");
       if(W_pion) legtitle41 =new TLegend(0.20,0.81,0.39,0.84," MVA_pion_weights/ Single particle");
       if(W_Kion) legtitle41 =new TLegend(0.20,0.81,0.39,0.84," MVA_Kion_weights/ Single particle");
       if(W_all) legtitle41 =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
       if(W_all_ecore) legtitle41 =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
   }
   legtitle41->SetTextSize(0.04);
   legtitle41->Draw();

//////////////////////////////////////////////////
   canv->cd(6);
   TPad *pad4all = new TPad("pad4all","pad4all",0,0,0.99,0.99);
   pad4all->Draw();    
   pad4all->cd();
   
   if(data_single) Ymin=100.0;//single
   if(data_embed) Ymin=20.0;//embed
   if(data_single) Ymax=180.0;//single
   if(data_embed) Ymax=60.0;//embed 
   /*
   if(data_single) Ymin=0.7;//single
   if(data_embed) Ymin=0.7;//embed
   if(data_single) Ymax=1.1;//single
   if(data_embed) Ymax=1.1;//embed  
   */
   Xmin=0.7;
   Xmax=1.1;  
  // gPad->SetLogy();
   hframe0=gPad->DrawFrame(Xmin,Ymin,Xmax,Ymax);
   hframe0->GetXaxis()->SetTitle("eID_efficiency"); 
   // hframe0->GetXaxis()->CenterTitle();	
  // hframe0->GetYaxis()->SetTitle("Hadron S/(S+B)");
   hframe0->GetYaxis()->SetTitle("Hadron(-) S/Sqrt(S+B)");
   // hframe0->GetYaxis()->CenterTitle();
   TGraph *gP41all,*gP51all,*gP61all,*gDNN4all;
   gP41all=new TGraph(7,efficiency_Sall_BDT,SBratio_all_BDT);
   gP41all->SetMarkerStyle(26);
   gP41all->SetMarkerColor(2);
   gP41all->SetLineColor(2);
   gP41all->SetLineStyle(1);
   gP41all->SetLineWidth(1.2);
   gP41all->SetMarkerSize(3.7);
   gP41all->Draw("pl");

   gP51all=new TGraph(6,efficiency_Sall_SVM,SBratio_all_SVM);
   gP51all->SetMarkerStyle(26);
   gP51all->SetMarkerColor(3);
   gP51all->SetLineColor(3);
   gP51all->SetLineStyle(1);
   gP51all->SetLineWidth(1.2);
   gP51all->SetMarkerSize(3.7);
   gP51all->Draw("pl");

   gP61all=new TGraphErrors(6,efficiency_Sall_LD,SBratio_all_LD);
   gP61all->SetMarkerStyle(26);
   gP61all->SetMarkerColor(4);
   gP61all->SetLineColor(4);
   gP61all->SetLineStyle(1);
   gP61all->SetLineWidth(1.2);
   gP61all->SetMarkerSize(3.7);
   gP61all->Draw("pl");

   gDNN4all=new TGraph(6,efficiency_Sall_DNN_CPU,SBratio_all_DNN_CPU);
   gDNN4all->SetMarkerStyle(26);
   gDNN4all->SetMarkerColor(6);
   gDNN4all->SetLineColor(6);
   gDNN4all->SetLineStyle(1);
   gDNN4all->SetLineWidth(1.2);
   gDNN4all->SetMarkerSize(3.7);
   gDNN4all->Draw("pl");

   TLegend *legP4all =new TLegend(0.30,0.25,0.50,0.50);
   legP4all->AddEntry(gP41all,"  BDT","lep");
   legP4all->AddEntry(gP51all,"  SVM","lep");
   legP4all->AddEntry(gP61all,"  LD","lep");
   legP4all->AddEntry(gDNN4all,"  DNN","lep");
   legP4all->Draw();

   TLegend *legtitle4all =new TLegend(0.20,0.86,0.63,0.90,"#it{#bf{sPHENIX}} Simulation");
   legtitle4all->SetTextSize(0.05);
   legtitle4all->Draw();
  
   TLegend *legtitle41all;
    if(data_embed){
       if(W_antiproton) legtitle41all =new TLegend(0.20,0.84,0.30,0.87," MVA_antiproton_weights/ Embed");
       if(W_pion) legtitle41all =new TLegend(0.20,0.84,0.30,0.87," MVA_pion_weights/ Embed");
       if(W_Kion) legtitle41all =new TLegend(0.20,0.84,0.30,0.87," MVA_Kion_weights/ Embed");
       if(W_all) legtitle41all =new TLegend(0.20,0.84,0.30,0.87,"  Embed");
   }
   if(data_single){
       if(W_antiproton) legtitle41all =new TLegend(0.20,0.81,0.39,0.84," MVA_antiproton_weights/ Single particle");
       if(W_pion) legtitle41all =new TLegend(0.20,0.81,0.39,0.84," MVA_pion_weights/ Single particle");
       if(W_Kion) legtitle41all =new TLegend(0.20,0.81,0.39,0.84," MVA_Kion_weights/ Single particle");
       if(W_all) legtitle41all =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
       if(W_all_ecore) legtitle41all =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
   }
   legtitle41all->SetTextSize(0.04);
   legtitle41all->Draw();

   canv->RedrawAxis();
   TString psname1;
   const char * output_plot_eID;
   if(data_embed){
       if(W_antiproton)output_plot_eID="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_embed_cutpt2_12_N/Rejection_MVA_antiproton_weights_embed";
       if(W_pion)output_plot_eID="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_embed_cutpt2_12_N/Rejection_MVA_pion_weights_embed";
       if(W_Kion)output_plot_eID="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_embed_cutpt2_12_N/Rejection_MVA_Kion_weights_embed";
       if(W_all)output_plot_eID="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_embed_cutpt2_12_N/Rejection_MVA_all_weights_embed";
   }
   if(data_single){
       if(W_antiproton)output_plot_eID="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_cutpt2_12_N/Rejection_MVA_antiproton_weights_single";
       if(W_pion)output_plot_eID="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_cutpt2_12_N/Rejection_MVA_pion_weights_single";
       if(W_Kion)output_plot_eID="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_cutpt2_12_N/Rejection_MVA_Kion_weights_single";
       if(W_all)output_plot_eID="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_cutpt2_12_N/Rejection_MVA_all_weights_single";
       if(W_all_ecore)output_plot_eID="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_cutpt2_12_N/Rejection_MVA_all_weights_single_ecore";
   }
   psname1=Form("%s.pdf",output_plot_eID);
   canv->Print(psname1);

//////////////////////////////////////////////plot2
TCanvas *canv2= new TCanvas("canv2","Cali Canvas",1800,2700);
   canv2->Divide(2,3);
   canv2->cd(1);
   TPad *pad21 = new TPad("pad21","pad21",0,0,0.99,0.99);
   pad21->Draw();    
   pad21->cd();
   //pad21->SetFrameLineWidth(2);
   //pad21->SetFrameLineColor(1);
   Ymin=-0.5;
   if(data_embed)Ymax=1.5;//embed
   if(data_single)Ymax=1.5;//single
   Xmin=0.6;
   Xmax=1.1; 
   //gPad->SetLogy();

   hframe0=gPad->DrawFrame(Xmin,Ymin,Xmax,Ymax);
   hframe0->GetXaxis()->SetTitle("eID_efficiency"); 
   // hframe0->GetXaxis()->CenterTitle();	
   hframe0->GetYaxis()->SetTitle("Cuts");
   // hframe0->GetYaxis()->CenterTitle();

  // TF1 *fBDT=new TF1("fBDT","[0]+[1]*x+[2]*x*x+[3]*x*x*x",0,20);
   TF1 *fBDT=new TF1("fBDT","[0]+[1]*log(x)+[2]*x*x+[3]*x*x*x",0,1);
   fBDT->SetParameters(1.0,1.0,1.0,1.0);
   fBDT->SetLineColor(2);
   fBDT->SetLineStyle(1);
   fBDT->SetLineWidth(2.0);

   TGraph *gBDT,*gSVM,*gLD,*gDNN_CPU;
   gBDT=new TGraphErrors(7,efficiency_electron_BDT,Ncut_BDT);
   gBDT->SetMarkerStyle(26);
   gBDT->SetMarkerColor(2);
   gBDT->SetLineColor(2);
   gBDT->SetLineStyle(1);
   gBDT->SetLineWidth(1.2);
   gBDT->SetMarkerSize(3.7);
   //gBDT->Fit("fBDT");
   gBDT->Draw("pl");

   gSVM=new TGraphErrors(6,efficiency_electron_SVM,Ncut_SVM);
   gSVM->SetMarkerStyle(26);
   gSVM->SetMarkerColor(3);
   gSVM->SetLineColor(3);
   gSVM->SetLineStyle(1);
   gSVM->SetLineWidth(1.2);
   gSVM->SetMarkerSize(3.7);
   gSVM->Draw("pl");

   gLD=new TGraphErrors(6,efficiency_electron_LD,Ncut_LD);
   gLD->SetMarkerStyle(26);
   gLD->SetMarkerColor(4);
   gLD->SetLineColor(4);
   gLD->SetLineStyle(1);
   gLD->SetLineWidth(1.2);
   gLD->SetMarkerSize(3.7);
   gLD->Draw("pl");

   gDNN_CPU=new TGraphErrors(6,efficiency_electron_DNN_CPU,Ncut_DNN_CPU);
   gDNN_CPU->SetMarkerStyle(26);
   gDNN_CPU->SetMarkerColor(6);
   gDNN_CPU->SetLineColor(6);
   gDNN_CPU->SetLineStyle(1);
   gDNN_CPU->SetLineWidth(1.2);
   gDNN_CPU->SetMarkerSize(3.7);
   gDNN_CPU->Draw("pl");

   TLegend *legPcut1 =new TLegend(0.20,0.20,0.45,0.45);
   legPcut1->AddEntry(gBDT,"  BDT","lep");
   legPcut1->AddEntry(gSVM,"  SVM","lep");
   legPcut1->AddEntry(gLD,"  LD","lep");
   legPcut1->AddEntry(gDNN_CPU,"  DNN","lep");
   legPcut1->Draw();

   TLegend *legtitlecut1 =new TLegend(0.20,0.86,0.63,0.90,"#it{#bf{sPHENIX}} Simulation");
   legtitlecut1->SetTextSize(0.05);
   legtitlecut1->Draw();
  
   TLegend *legtitlecut;
   if(data_embed){
       if(W_antiproton) legtitlecut =new TLegend(0.20,0.84,0.30,0.87," MVA_antiproton_weights/ Embed");
       if(W_pion) legtitlecut =new TLegend(0.20,0.84,0.30,0.87," MVA_pion_weights/ Embed");
       if(W_Kion) legtitlecut =new TLegend(0.20,0.84,0.30,0.87," MVA_Kion_weights/ Embed");
       if(W_all) legtitlecut =new TLegend(0.20,0.84,0.30,0.87,"  Embed");
   }
   if(data_single){
       if(W_antiproton) legtitlecut =new TLegend(0.20,0.81,0.39,0.84," MVA_antiproton_weights/ Single particle");
       if(W_pion) legtitlecut =new TLegend(0.20,0.81,0.39,0.84," MVA_pion_weights/ Single particle");
       if(W_Kion) legtitlecut =new TLegend(0.20,0.81,0.39,0.84," MVA_Kion_weights/ Single particle");
       if(W_all) legtitlecut =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
       if(W_all_ecore) legtitlecut =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
   }
   legtitlecut->SetTextSize(0.04);
   legtitlecut->Draw();

////////////////////////////////////////////////////
   canv2->cd(2);
   TPad *pad21all = new TPad("pad21all","pad21all",0,0,0.99,0.99);
   pad21all->Draw();    
   pad21all->cd();
   //pad21all->SetFrameLineWidth(2);
   //pad21all->SetFrameLineColor(1);

   Ymin=0.55;
   if(data_embed)Ymax=1.4;//embed
   if(data_single)Ymax=1.4;//single
   Xmin=0.0;
   Xmax=14.0;  

    hframe0=gPad->DrawFrame(Xmin,Ymin,Xmax,Ymax);
    hframe0->GetXaxis()->SetTitle("p_{t} (GeV)"); 
//  hframe0->GetXaxis()->CenterTitle();	
    hframe0->GetYaxis()->SetTitle("Electron ID efficiency (100%)");
//  hframe0->GetYaxis()->CenterTitle();


   /////////////
  TGraphErrors *gN2,*gN21,*gN22;  
  gN22=new TGraphErrors(6,aa_pt_N,cc_pt_N_SVM,err_aa_pt_N,err_cc_pt_N_SVM);
  gN22->SetMarkerStyle(26);
  gN22->SetMarkerColor(3);
  gN22->SetLineColor(3);
  gN22->SetLineStyle(1);
  gN22->SetLineWidth(1.2);
  gN22->SetMarkerSize(3.6);
  gN22->Draw("p");
  gN21=new TGraphErrors(6,aa_pt_N,cc_pt_N_BDT,err_aa_pt_N,err_cc_pt_N_BDT);
  gN21->SetMarkerStyle(24);
  gN21->SetMarkerColor(2);
  gN21->SetLineColor(2);
  gN21->SetLineStyle(1);
  gN21->SetLineWidth(1.2);
  gN21->SetMarkerSize(3.6);
  gN21->Draw("p");
  gN2=new TGraphErrors(6,aa_pt_N,cc_pt_N_cuts,err_aa_pt_N,err_cc_pt_N_cuts);
  gN2->SetMarkerStyle(27);
  gN2->SetMarkerColor(4);
  gN2->SetLineColor(4);
  gN2->SetLineStyle(1);
  gN2->SetLineWidth(1.2);
  gN2->SetMarkerSize(3.6);
  gN2->Draw("p");
  
  TLegend *leg2N =new TLegend(0.65,0.60,0.70,0.85);
  leg2N->AddEntry(gN21,"  BDT","lep");
  leg2N->AddEntry(gN22,"  SVM","lep");
  leg2N->AddEntry(gN2,"  Trad. cuts","lep");
  leg2N->Draw();
  /////////////////////
  
   TLegend *legtitlecut1all =new TLegend(0.20,0.86,0.63,0.90,"#it{#bf{sPHENIX}} Simulation");
   legtitlecut1all->SetTextSize(0.05);
   legtitlecut1all->Draw();
  
   TLegend *legtitlecutall;
   if(data_embed){
       if(W_antiproton) legtitlecutall =new TLegend(0.20,0.84,0.30,0.87," MVA_antiproton_weights/ Embed");
       if(W_pion) legtitlecutall =new TLegend(0.20,0.84,0.30,0.87," MVA_pion_weights/ Embed");
       if(W_Kion) legtitlecutall =new TLegend(0.20,0.84,0.30,0.87," MVA_Kion_weights/ Embed");
       if(W_all) legtitlecutall =new TLegend(0.20,0.81,0.39,0.84,"  Embed");
   }
   if(data_single){
       if(W_antiproton) legtitlecutall =new TLegend(0.20,0.81,0.39,0.84," MVA_antiproton_weights/ Single particle");
       if(W_pion) legtitlecutall =new TLegend(0.20,0.81,0.39,0.84," MVA_pion_weights/ Single particle");
       if(W_Kion) legtitlecutall =new TLegend(0.20,0.81,0.39,0.84," MVA_Kion_weights/ Single particle");
       if(W_all) legtitlecutall =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
       if(W_all_ecore) legtitlecutall =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
   }
   legtitlecutall->SetTextSize(0.040);
   legtitlecutall->Draw();

/////////////////////////////////////////////////////////////////////////////////////////////////////
   canv2->cd(3);
   TPad *pad31all = new TPad("pad31all","pad31all",0,0,0.99,0.99);
   pad31all->Draw();    
   pad31all->cd();
   //pad31all->SetFrameLineWidth(2);
   //pad31all->SetFrameLineColor(1);
   Ymin=1.0;
   if(data_embed)Ymax=10000;//embed
   if(data_single)Ymax=10000;//single
   Xmin=0.0;
   if(data_embed)Xmax=14.0; 
   if(data_single)Xmax=14.0; 
   gPad->SetLogy();
   int npoint=0;
   if(data_embed)npoint=6; 
   if(data_single)npoint=6; 

   hframe0=gPad->DrawFrame(Xmin,Ymin,Xmax,Ymax);
   hframe0->GetXaxis()->SetTitle("pt (GeV)"); 
   // hframe0->GetXaxis()->CenterTitle();	
  // if(data_single) hframe0->GetYaxis()->SetTitle("hadron(-) rejection at 90% eID efficiency");
   if(data_single) hframe0->GetYaxis()->SetTitle("#pi^{-} rejection at 90% eID efficiency"); //"#pi^{-}+K^{-}+#bar{p}"

   //if(data_embed) hframe0->GetYaxis()->SetTitle("hadron(-) rejection at 90% eID efficiency");
   if(data_embed) hframe0->GetYaxis()->SetTitle("#bar{p} rejection at 90% eID efficiency");
   // hframe0->GetYaxis()->CenterTitle();

   double efficiency_electron_cuts_tt=0.0,efficiency_electron_BDT_tt=0.0,efficiency_electron_SVM_tt=0.0;

   if(Nelectron_cuts>0 & Nelectron>0) efficiency_electron_cuts_tt=1.0*Nelectron_cuts/Nelectron; //traditional cuts
   if(Nelectron_cuts>0 & Nelectron>0) efficiency_electron_BDT_tt=1.0*Nelectron_BDT/Nelectron; 
   if(Nelectron_cuts>0 & Nelectron>0) efficiency_electron_SVM_tt=1.0*Nelectron_SVM/Nelectron; 
   std::cout<<Nelectron<<"; "<<Nelectron_cuts<<"; efficiency_electron_cuts_tt= "<<efficiency_electron_cuts_tt<<std::endl;
   std::cout<<Nelectron<<"; "<<Nelectron_BDT<<"; efficiency_electron_BDT_tt= "<<efficiency_electron_BDT_tt<<std::endl;
   std::cout<<Nelectron<<"; "<<Nelectron_SVM<<"; efficiency_electron_SVM_tt= "<<efficiency_electron_SVM_tt<<std::endl;
  // rejection_all_SVM_pt[4]=60000; //for antiproton
  // rejection_all_BDT_pt[4]=60000; //for antiproton
   TGraphErrors *gPall_BDT_pt,*gPall_SVM_pt,*gPall_LD_pt,*gPall_DNN_pt,*gPall_cuts_pt;
   gPall_SVM_pt=new TGraphErrors(npoint,Npt,rejection_all_SVM_pt,err_Npt,err_rejection_all_SVM_pt); //single
  // gPall_SVM_pt=new TGraphErrors(5,Npt,rejection_all_SVM_pt,err_Npt,err_rejection_all_SVM_pt); //embed
   gPall_SVM_pt->SetMarkerStyle(26);
   gPall_SVM_pt->SetMarkerColor(3);
   gPall_SVM_pt->SetLineColor(3);
   gPall_SVM_pt->SetLineStyle(1);
   gPall_SVM_pt->SetLineWidth(1.2);
   gPall_SVM_pt->SetMarkerSize(3.6);
   gPall_SVM_pt->Draw("pl");

   gPall_BDT_pt=new TGraphErrors(npoint,Npt,rejection_all_BDT_pt,err_Npt,err_rejection_all_BDT_pt); //single
   gPall_BDT_pt->SetMarkerStyle(24);
   gPall_BDT_pt->SetMarkerColor(2);
   gPall_BDT_pt->SetLineColor(2);
   gPall_BDT_pt->SetLineStyle(1);
   gPall_BDT_pt->SetLineWidth(1.2);
   gPall_BDT_pt->SetMarkerSize(3.6);
   gPall_BDT_pt->Draw("pl");

   gPall_cuts_pt=new TGraphErrors(npoint,Npt,rejection_all_cuts_pt,err_Npt,err_rejection_all_cuts_pt); //single
   gPall_cuts_pt->SetMarkerStyle(27);
   gPall_cuts_pt->SetMarkerColor(4);
   gPall_cuts_pt->SetLineColor(4);
   gPall_cuts_pt->SetLineStyle(1);
   gPall_cuts_pt->SetLineWidth(1.2);
   gPall_cuts_pt->SetMarkerSize(3.8);
   gPall_cuts_pt->Draw("pl");

   TLegend *legall_pt =new TLegend(0.65,0.20,0.70,0.45);
   legall_pt->AddEntry(gPall_BDT_pt,"  BDT","lep");
   legall_pt->AddEntry(gPall_SVM_pt,"  SVM","lep");
  // legall_pt->AddEntry(gPall_LD_pt,"  LD","lep");
  // legall_pt->AddEntry(gPall_DNN_pt,"  DNN","lep");
   legall_pt->AddEntry(gPall_cuts_pt,"  Trad. cuts","lep");
   legall_pt->Draw();

   TLegend *legtitleall_pt =new TLegend(0.20,0.86,0.63,0.90,"#it{#bf{sPHENIX}} Simulation");
   legtitleall_pt->SetTextSize(0.05);
   legtitleall_pt->Draw();
  
   TLegend *legtitlecutall_pt;
   if(data_embed){
       if(W_antiproton) legtitlecutall_pt =new TLegend(0.20,0.81,0.30,0.84," MVA_antiproton_weights/ Embed");
       if(W_pion) legtitlecutall_pt =new TLegend(0.20,0.81,0.30,0.84," MVA_pion_weights/ Embed");
       if(W_Kion) legtitlecutall_pt =new TLegend(0.20,0.81,0.30,0.84," MVA_Kion_weights/ Embed");
       if(W_all) legtitlecutall_pt =new TLegend(0.20,0.81,0.30,0.84,"  Embed");
   }
   if(data_single){
       if(W_antiproton) legtitlecutall_pt =new TLegend(0.20,0.81,0.39,0.84," MVA_antiproton_weights/ Single particle");
       if(W_pion) legtitlecutall_pt =new TLegend(0.20,0.81,0.39,0.84," MVA_pion_weights/ Single particle");
       if(W_Kion) legtitlecutall_pt =new TLegend(0.20,0.81,0.39,0.84," MVA_Kion_weights/ Single particle");
       if(W_all) legtitlecutall_pt =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
       if(W_all_ecore) legtitlecutall_pt =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
   }
   legtitlecutall_pt->SetTextSize(0.04);
   legtitlecutall_pt->Draw();

   /////////////////////////////////////////////////////////////////////////////////////////////////////
   canv2->cd(4);
   TPad *pad41all = new TPad("pad41all","pad41all",0,0,0.99,0.99);
   pad41all->Draw();    
   pad41all->cd();
   //pad41all->SetFrameLineWidth(2);
   //pad41all->SetFrameLineColor(1);
   Ymin=0.0001;
   if(data_embed)Ymax=0.5;//embed
   if(data_single)Ymax=0.5;//single
   Xmin=0.0;
   if(data_embed)Xmax=14.0; 
   if(data_single)Xmax=14.0; 
   gPad->SetLogy();
   int npoint_inverse=0;
   if(data_embed)npoint_inverse=6; 
   if(data_single)npoint_inverse=6; 

   hframe0=gPad->DrawFrame(Xmin,Ymin,Xmax,Ymax);
   hframe0->GetXaxis()->SetTitle("pt (GeV)"); 
   // hframe0->GetXaxis()->CenterTitle();	
  // if(data_single) hframe0->GetYaxis()->SetTitle("hadron(-) inv. rejection factor at 90% eID eff.");
   if(data_single) hframe0->GetYaxis()->SetTitle("#pi^{-} inv. rejection factor at 90% eID eff."); //"#pi^{-}+K^{-}+#bar{p}"

   //if(data_embed) hframe0->GetYaxis()->SetTitle("hadron(-) inv. rejection factor at 90% eID eff.");
   if(data_embed) hframe0->GetYaxis()->SetTitle("#bar{p} inv. rejection factor at 90% eID eff.");
   // hframe0->GetYaxis()->CenterTitle();

   TGraphErrors *gPall_BDT_pt_inverse,*gPall_SVM_pt_inverse,*gPall_LD_pt_inverse,*gPall_DNN_pt_inverse,*gPall_cuts_pt_inverse;
   gPall_SVM_pt_inverse=new TGraphErrors(npoint_inverse,Npt,rejection_all_SVM_pt_inverse,err_Npt,err_rejection_all_SVM_pt_inverse); //single
  // gPall_SVM_pt_inverse=new TGraphErrors(5,Npt,rejection_all_SVM_pt_inverse,err_Npt,err_rejection_all_SVM_pt_inverse); //embed
   gPall_SVM_pt_inverse->SetMarkerStyle(26);
   gPall_SVM_pt_inverse->SetMarkerColor(3);
   gPall_SVM_pt_inverse->SetLineColor(3);
   gPall_SVM_pt_inverse->SetLineStyle(1);
   gPall_SVM_pt_inverse->SetLineWidth(1.2);
   gPall_SVM_pt_inverse->SetMarkerSize(3.6);
   gPall_SVM_pt_inverse->Draw("pl");

   gPall_BDT_pt_inverse=new TGraphErrors(npoint_inverse,Npt,rejection_all_BDT_pt_inverse,err_Npt,err_rejection_all_BDT_pt_inverse); //single
   gPall_BDT_pt_inverse->SetMarkerStyle(24);
   gPall_BDT_pt_inverse->SetMarkerColor(2);
   gPall_BDT_pt_inverse->SetLineColor(2);
   gPall_BDT_pt_inverse->SetLineStyle(1);
   gPall_BDT_pt_inverse->SetLineWidth(1.2);
   gPall_BDT_pt_inverse->SetMarkerSize(3.6);
   gPall_BDT_pt_inverse->Draw("pl");

   gPall_cuts_pt_inverse=new TGraphErrors(npoint_inverse,Npt,rejection_all_cuts_pt_inverse,err_Npt,err_rejection_all_cuts_pt_inverse); //single
   gPall_cuts_pt_inverse->SetMarkerStyle(27);
   gPall_cuts_pt_inverse->SetMarkerColor(4);
   gPall_cuts_pt_inverse->SetLineColor(4);
   gPall_cuts_pt_inverse->SetLineStyle(1);
   gPall_cuts_pt_inverse->SetLineWidth(1.2);
   gPall_cuts_pt_inverse->SetMarkerSize(3.8);
   gPall_cuts_pt_inverse->Draw("pl");

   TLegend *legall_pt_inverse =new TLegend(0.65,0.60,0.70,0.85);
   legall_pt_inverse->AddEntry(gPall_BDT_pt_inverse,"  BDT","lep");
   legall_pt_inverse->AddEntry(gPall_SVM_pt_inverse,"  SVM","lep");
  // legall_pt_inverse->AddEntry(gPall_LD_pt_inverse,"  LD","lep");
  // legall_pt_inverse->AddEntry(gPall_DNN_pt_inverse,"  DNN","lep");
   legall_pt_inverse->AddEntry(gPall_cuts_pt_inverse,"  Trad. cuts","lep");
   legall_pt_inverse->Draw();

   TLegend *legtitleall_pt_inverse =new TLegend(0.20,0.86,0.63,0.90,"#it{#bf{sPHENIX}} Simulation");
   legtitleall_pt_inverse->SetTextSize(0.05);
   legtitleall_pt_inverse->Draw();
  
   TLegend *legtitlecutall_pt_inverse;
   if(data_embed){
       if(W_antiproton) legtitlecutall_pt_inverse =new TLegend(0.20,0.81,0.30,0.84," MVA_antiproton_weights/ Embed");
       if(W_pion) legtitlecutall_pt_inverse =new TLegend(0.20,0.81,0.30,0.84," MVA_pion_weights/ Embed");
       if(W_Kion) legtitlecutall_pt_inverse =new TLegend(0.20,0.81,0.30,0.84," MVA_Kion_weights/ Embed");
       if(W_all) legtitlecutall_pt_inverse =new TLegend(0.20,0.81,0.30,0.84,"  Embed");
   }
   if(data_single){
       if(W_antiproton) legtitlecutall_pt_inverse =new TLegend(0.20,0.81,0.39,0.84," MVA_antiproton_weights/ Single particle");
       if(W_pion) legtitlecutall_pt_inverse =new TLegend(0.20,0.81,0.39,0.84," MVA_pion_weights/ Single particle");
       if(W_Kion) legtitlecutall_pt_inverse =new TLegend(0.20,0.81,0.39,0.84," MVA_Kion_weights/ Single particle");
       if(W_all) legtitlecutall_pt_inverse =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
       if(W_all) legtitlecutall_pt_inverse =new TLegend(0.20,0.71,0.39,0.74,"  At 90% eID efficiency");
       if(W_all_ecore) legtitlecutall_pt_inverse =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
   }
   legtitlecutall_pt->SetTextSize(0.045);
   legtitlecutall_pt->Draw();

/////////////////////////////////////////////////////////////////////////////////////////////////////
   canv2->cd(5);
   TPad *pad51all = new TPad("pad51all","pad51all",0,0,0.99,0.99);
   pad51all->Draw();    
   pad51all->cd();
   //pad51all->SetFrameLineWidth(2);
   //pad51all->SetFrameLineColor(1);
   Ymin=1.0;
   if(data_embed)Ymax=100000;//embed
   if(data_single)Ymax=100000;//single
   Xmin=0.0;
   if(data_embed)Xmax=20.0; 
   if(data_single)Xmax=20.0; 
   gPad->SetLogy();

   if(data_embed)npoint=5; 
   if(data_single)npoint=5; 

   hframe0=gPad->DrawFrame(Xmin,Ymin,Xmax,Ymax);
   hframe0->GetXaxis()->SetTitle("b (fm)"); 
   // hframe0->GetXaxis()->CenterTitle();	
   //if(data_single) hframe0->GetYaxis()->SetTitle("hadron(-) rejection at 90% eID efficiency");
   if(data_single) hframe0->GetYaxis()->SetTitle("#pi^{-} rejection at 90% eID efficiency"); //"#pi^{-}+K^{-}+#bar{p}"

   //if(data_embed) hframe0->GetYaxis()->SetTitle("hadron(-) rejection at 90% eID efficiency");
   if(data_embed) hframe0->GetYaxis()->SetTitle("#bar{p} rejection at 90% eID efficiency");
   // hframe0->GetYaxis()->CenterTitle();

   TGraphErrors *gPall_BDT_bimp,*gPall_SVM_bimp,*gPall_LD_bimp,*gPall_DNN_bimp,*gPall_cuts_bimp;
   gPall_SVM_bimp=new TGraphErrors(npoint,Nbimp,rejection_all_SVM_bimp,err_Nbimp,err_rejection_all_SVM_bimp); //single
  // gPall_SVM_bimp=new TGraphErrors(5,Nbimp,rejection_all_SVM_bimp,err_Nbimp,err_rejection_all_SVM_bimp); //embed
   gPall_SVM_bimp->SetMarkerStyle(26);
   gPall_SVM_bimp->SetMarkerColor(3);
   gPall_SVM_bimp->SetLineColor(3);
   gPall_SVM_bimp->SetLineStyle(1);
   gPall_SVM_bimp->SetLineWidth(1.2);
   gPall_SVM_bimp->SetMarkerSize(3.6);
   gPall_SVM_bimp->Draw("pl");

   gPall_BDT_bimp=new TGraphErrors(npoint,Nbimp,rejection_all_BDT_bimp,err_Nbimp,err_rejection_all_BDT_bimp); //single
   gPall_BDT_bimp->SetMarkerStyle(24);
   gPall_BDT_bimp->SetMarkerColor(2);
   gPall_BDT_bimp->SetLineColor(2);
   gPall_BDT_bimp->SetLineStyle(1);
   gPall_BDT_bimp->SetLineWidth(1.2);
   gPall_BDT_bimp->SetMarkerSize(3.6);
   gPall_BDT_bimp->Draw("pl");

   gPall_cuts_bimp=new TGraphErrors(npoint,Nbimp,rejection_all_cuts_bimp,err_Nbimp,err_rejection_all_cuts_bimp); //single
   gPall_cuts_bimp->SetMarkerStyle(27);
   gPall_cuts_bimp->SetMarkerColor(4);
   gPall_cuts_bimp->SetLineColor(4);
   gPall_cuts_bimp->SetLineStyle(1);
   gPall_cuts_bimp->SetLineWidth(1.2);
   gPall_cuts_bimp->SetMarkerSize(3.8);
   gPall_cuts_bimp->Draw("pl");

   TLegend *legall_bimp =new TLegend(0.65,0.20,0.70,0.45);
   legall_bimp->AddEntry(gPall_BDT_bimp,"  BDT","lep");
   legall_bimp->AddEntry(gPall_SVM_bimp,"  SVM","lep");
  // legall_bimp->AddEntry(gPall_LD_bimp,"  LD","lep");
  // legall_bimp->AddEntry(gPall_DNN_bimp,"  DNN","lep");
   legall_bimp->AddEntry(gPall_cuts_bimp,"  Trad. cuts","lep");
   legall_bimp->Draw();

   TLegend *legtitleall_bimp =new TLegend(0.20,0.86,0.63,0.90,"#it{#bf{sPHENIX}} Simulation");
   legtitleall_bimp->SetTextSize(0.05);
   legtitleall_bimp->Draw();
  
   TLegend *legtitlecutall_bimp;
   if(data_embed){
       if(W_antiproton) legtitlecutall_bimp =new TLegend(0.20,0.81,0.30,0.84," MVA_antiproton_weights/ Embed");
       if(W_pion) legtitlecutall_bimp =new TLegend(0.20,0.81,0.30,0.84," MVA_pion_weights/ Embed");
       if(W_Kion) legtitlecutall_bimp =new TLegend(0.20,0.81,0.30,0.84," MVA_Kion_weights/ Embed");
       if(W_all) legtitlecutall_bimp =new TLegend(0.20,0.81,0.30,0.84,"  Embed");
   }
   if(data_single){
       if(W_antiproton) legtitlecutall_bimp =new TLegend(0.20,0.81,0.39,0.84," MVA_antiproton_weights/ Single particle");
       if(W_pion) legtitlecutall_bimp =new TLegend(0.20,0.81,0.39,0.84," MVA_pion_weights/ Single particle");
       if(W_Kion) legtitlecutall_bimp =new TLegend(0.20,0.81,0.39,0.84," MVA_Kion_weights/ Single particle");
       if(W_all) legtitlecutall_bimp =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
       if(W_all_ecore) legtitlecutall_bimp =new TLegend(0.20,0.81,0.39,0.84,"  Single particle");
   }
   legtitlecutall_bimp->SetTextSize(0.045);
   legtitlecutall_bimp->Draw();
//////////////////////////////////////////////////////////////////

   canv2->RedrawAxis();
   TString psnamecut1;
   const char * output_plot_eID_cut;
   if(data_embed){
       if(W_antiproton)output_plot_eID_cut="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_embed_cutpt2_12_N/cuts_on_response/Rejection_MVA_antiproton_weights_embed";
       if(W_pion)output_plot_eID_cut="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_embed_cutpt2_12_N/cuts_on_response/Rejection_MVA_pion_weights_embed";
       if(W_Kion)output_plot_eID_cut="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_embed_cutpt2_12_N/cuts_on_response/Rejection_MVA_Kion_weights_embed";
       if(W_all)output_plot_eID_cut="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_embed_cutpt2_12_N/cuts_on_response/Rejection_MVA_all_weights_embed";
   }
   if(data_single){
       if(W_antiproton)output_plot_eID_cut="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_cutpt2_12_N/cuts_on_response/Rejection_MVA_antiproton_weights_single";
       if(W_pion)output_plot_eID_cut="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_cutpt2_12_N/cuts_on_response/Rejection_MVA_pion_weights_single";
       if(W_Kion)output_plot_eID_cut="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_cutpt2_12_N/cuts_on_response/Rejection_MVA_Kion_weights_single";
       if(W_all)output_plot_eID_cut="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_cutpt2_12_N/cuts_on_response/Rejection_MVA_all_weights_single";
       if(W_all_ecore)output_plot_eID_cut="/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification/TMVA_App_eID_cutpt2_12_N/cuts_on_response/Rejection_MVA_all_weights_single_ecore";
   }
   psnamecut1=Form("%s.pdf",output_plot_eID_cut);
   canv2->Print(psnamecut1);

   //track efficiency
   double eff1=1.0*N_track/N_raw;  //good tracks efficiency
   double eff2=1.0*N_track_pt2/N_raw; //pt>2.0 GeV cut efficiency after for all tracks
   double eff3=1.0*N_track_pt2/N_track; //pt>2.0 GeV cut efficiency after for good tracks
   std::cout<<eff1<<"; "<<eff2<<"; "<<eff3<<std::endl;

//////////////////////////////////////////////////
}
void TMVAClassificationTraining_Test(){

    std::cout <<"OK!!!!!!!!!!!!!!!!!!!!!"<< std::endl;
}

int main( int argc, char** argv )
{
   TString methodList;
   for (int i=1; i<argc; i++) {
      TString regMethod(argv[i]);
      if(regMethod=="-b" || regMethod=="--batch") continue;
      if (!methodList.IsNull()) methodList += TString(",");
      methodList += regMethod;
   }
   TMVAClassificationApplication_eID_N(methodList);
   TMVAClassificationTraining_Test();
   return 0;
}

