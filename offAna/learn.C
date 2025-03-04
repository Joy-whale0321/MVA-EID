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

// #include "sPhenixStyle.h"
// #include "sPhenixStyle.C"

using namespace TMVA;

void TMVAClassificationApplication_eID_N( TString myMethodList = "" )
{
    //---------------------------------------------------------------
    // This loads the library

    // Default MVA methods to be trained + tested
    std::map<std::string,int> Use;

    // Boosted Decision Trees
    Use["BDT"]             = 1; // uses Adaptive Boost
    Use["BDTG"]            = 0; // uses Gradient Boost
    Use["BDTB"]            = 0; // uses Bagging
    Use["BDTD"]            = 0; // decorrelation + Adaptive Boost
    Use["BDTF"]            = 0; // allow usage of fisher discriminant for node splitting

    std::cout << std::endl;
    std::cout << "==> Start TMVAClassificationApplication" << std::endl;

    // --------------------------------------------------------------------------------------------------

    // Create the Reader object

    TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );

    // Create a set of variables and declare them to the reader
    // - the variable names MUST corresponds in name and type to those given in the weight file(s) used
    Float_t var1, var2;
    Float_t var3, var4;
    Float_t var5, var6;
    reader->AddVariable( "var1", &var1 );
    reader->AddVariable( "var2", &var2 );
    reader->AddVariable( "var3", &var3 );

    // Spectator variables declared in the training have to be added to the reader, too
    Float_t spec1,spec2;
    reader->AddSpectator( "spec1 := var1*2",   &spec1 );
    reader->AddSpectator( "spec2 := var1*3",   &spec2 );

    Float_t Category_cat1, Category_cat2, Category_cat3;
    if (Use["Category"])
    {
        // Add artificial spectators for distinguishing categories
        reader->AddSpectator( "Category_cat1 := (var3<=0)",   &Category_cat1 );
        reader->AddSpectator( "Category_cat2 := (var3>0)",  &Category_cat2 );
        reader->AddSpectator( "Category_cat3 := (var3>0)",  &Category_cat3 );
    }

    bool W_all=true;
    bool W_all_ecore=false;
    bool W_allN=false;
    bool W_antiproton=false;
    bool W_pion=false;
    bool W_Kion=false;

    bool data_single=true;
    bool data_embed=false;

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
    for (std::map<std::string,int>::iterator it = Use.begin(); it != Use.end(); it++) 
    {
        if (it->second) 
        {
            TString methodName = TString(it->first) + TString(" method"); // ？method 后缀？
            TString weightfile = dir + prefix + TString("_") + TString(it->first) + TString(".weights.xml");
            reader->BookMVA( methodName, weightfile );
        }
    }

    // Book output histograms
    TH1F *histBdt = nullptr;
    TH1F *histBdtG = nullptr;
    TH1F *histBdtB = nullptr;
    TH1F *histBdtD = nullptr;
    TH1F *histBdtF = nullptr;


    TH1F *h1electron_BDT = nullptr;
    TH1F *h1Sall_BDT = nullptr;
    TH1F *h1background_BDT = nullptr;
    TH1F *h1background_pion_BDT = nullptr;
    TH1F *h1background_antiproton_BDT = nullptr;
    TH1F *h1background_all_BDT = nullptr;
    h1electron_BDT = new TH1F("h1electron_BDT", "h1electron_BDT", nbin, -1.0, 1.0); //weihu
    h1Sall_BDT = new TH1F("h1Sall_BDT", "h1Sall_BDT", nbin, -1.0, 1.0); //weihu
    h1background_BDT = new TH1F("h1background_BDT", "h1background_BDT", nbin, -1.0, 1.0); //weihu
    h1background_pion_BDT = new TH1F("h1background_pion_BDT", "h1background_pion_BDT", nbin, -1.0, 1.0); //weihu
    h1background_antiproton_BDT = new TH1F("h1background_antiproton_BDT", "h1background_antiproton_BDT", nbin, -1.0, 1.0); //weihu
    h1background_all_BDT = new TH1F("h1background_all_BDT", "h1background_all_BDT", nbin, -1.0, 1.0); //weihu

    TH1F *h1EOP = nullptr;    //E3x3/p
    TH1F *h1EOP_e = nullptr;  //E3x3/p
    TH1F *h1EOP_cut = nullptr; //E3x3/p
    TH1F *h1EcOP = nullptr;   //Ecore/p
    h1EOP = new TH1F("h1EOP", "h1EOP", nbin, 0.0, 5.0);
    h1EOP_e = new TH1F("h1EOP_e", "h1EOP_e", 50, 0.0, 2.0);
    h1EOP_cut = new TH1F("h1EOP_cut", "h1EOP_cut", nbin, 0.0, 5.0);
    h1EcOP = new TH1F("h1EcOP", "h1EcOP", nbin, 0.0, 5.0);

    TH1F *h1HOM = nullptr;    //inH3x3/E3x3
    TH1F *h1HOM_e = nullptr;  //inH3x3/E3x3
    TH1F *h1CEMCchi2 = nullptr;    //CEMC cluster Chi2
    TH1F *h1CEMCchi2_e = nullptr;  //CEMC cluster Chi2
    h1HOM = new TH1F("h1HOM", "h1HOM", nbin, 0.0, 5.0);
    h1HOM_e = new TH1F("h1HOM_e", "h1HOM_e", nbin, 0.0, 5.0);
    h1CEMCchi2 = new TH1F("h1CEMCchi2", "h1CEMCchi2", nbin, 0.0, 20.0);
    h1CEMCchi2_e = new TH1F("h1CEMCchi2_e", "h1CEMCchi2_e", nbin, 0.0, 20.0);

    TH1F *h1pt = nullptr;
    TH1F *h1pt_cut = nullptr;
    h1pt = new TH1F("h1pt", "h1pt", nbin, 0.0, 20.0);
    h1pt_cut = new TH1F("h1pt_cut", "h1pt_cut", nbin, 0.0, 20.0);

    TH1F *h1flavor_1 = nullptr;
    TH1F *h1flavor_2 = nullptr;
    h1flavor_1 = new TH1F("h1flavor_1", "h1flavor_1", 3000, -3000.0, 3000.0);
    h1flavor_2 = new TH1F("h1flavor_2", "h1flavor_2", 3000, -3000.0, 3000.0);

    TH1F *h1var1_EOP_1 = nullptr;
    TH1F *h1var2_HOM_1 = nullptr;
    TH1F *h1var3_Chi2_1 = nullptr;
    TH1F *h1var1_EOP_2 = nullptr;
    TH1F *h1var2_HOM_2 = nullptr;
    TH1F *h1var3_Chi2_2 = nullptr;
    h1var1_EOP_1 = new TH1F("h1var1_EOP_1", "h1var1_EOP_1", 30, 0.0, 3.0);
    h1var2_HOM_1 = new TH1F("h1var2_HOM_1", "h1var2_HOM_1", 30, 0.0, 3.0);
    h1var3_Chi2_1 = new TH1F("h1var3_Chi2_1", "h1var3_Chi2_1", 100, 0.0, 10.0);
    h1var1_EOP_2 = new TH1F("h1var1_EOP_2", "h1var1_EOP_2", 30, 0.0, 3.0);
    h1var2_HOM_2 = new TH1F("h1var2_HOM_2", "h1var2_HOM_2", 30, 0.0, 3.0);
    h1var3_Chi2_2 = new TH1F("h1var3_Chi2_2", "h1var3_Chi2_2", 100, 0.0, 10.0);

    TH1F *h1_p_1 = nullptr;
    TH1F *h1_pt_1 = nullptr;
    TH1F *h1_Eemcal3x3_1 = nullptr;
    TH1F *h1_p_2 = nullptr;
    TH1F *h1_pt_2 = nullptr;
    TH1F *h1_Eemcal3x3_2 = nullptr;
    h1_p_1 = new TH1F("h1_p_1", "h1_p_1", 100, 1.5, 49.5);
    h1_pt_1 = new TH1F("h1_pt_1", "h1_pt_1", 100, 1.5, 29.5);
    h1_Eemcal3x3_1 = new TH1F("h1_Eemcal3x3_1", "h1_Eemcal3x3_1", 180, 1.5, 19.5);
    h1_p_2 = new TH1F("h1_p_2", "h1_p_2", 100, 1.5, 49.5);
    h1_pt_2 = new TH1F("h1_pt_2", "h1_pt_2", 100, 1.5, 29.5);
    h1_Eemcal3x3_2 = new TH1F("h1_Eemcal3x3_2", "h1_Eemcal3x3_2", 180, 1.5, 19.5);

    TH2F *h2_reponse_pt = nullptr;
    TH2F *h2_reponse_EOP = nullptr;
    TH2F *h2_reponse_HOM = nullptr;
    TH2F *h2_reponse_chi2 = nullptr;
    h2_reponse_pt = new TH2F("h2_reponse_pt", "h2_reponse_pt", 50, -0.5, 0.5, 100, 1.5, 12.5);
    h2_reponse_EOP = new TH2F("h2_reponse_EOP", "h2_reponse_EOP", 50, -0.5, 0.5, 40, 0.0, 4.0);
    h2_reponse_HOM = new TH2F("h2_reponse_HOM", "h2_reponse_HOM", 50, -0.5, 0.5, 100, 0.0, 1.0);
    h2_reponse_chi2 = new TH2F("h2_reponse_chi2", "h2_reponse_chi2", 50, -0.5, 0.5, 200, 0.0, 20.0);

    if (Use["Cuts"])          histCuts    = new TH1F("MVA_Cuts", "MVA_Cuts", nbin, -2, 4); //weihu
    if (Use["Likelihood"])    histLk      = new TH1F("MVA_Likelihood", "MVA_Likelihood", nbin, -1, 1);
    if (Use["LikelihoodD"])   histLkD     = new TH1F("MVA_LikelihoodD", "MVA_LikelihoodD", nbin, -1, 0.9999);
    if (Use["LikelihoodPCA"]) histLkPCA   = new TH1F("MVA_LikelihoodPCA", "MVA_LikelihoodPCA", nbin, -1, 1);
    if (Use["LikelihoodKDE"]) histLkKDE   = new TH1F("MVA_LikelihoodKDE", "MVA_LikelihoodKDE", nbin, -0.00001, 0.99999);
    if (Use["LikelihoodMIX"]) histLkMIX   = new TH1F("MVA_LikelihoodMIX", "MVA_LikelihoodMIX", nbin, 0, 1);
    if (Use["PDERS"])         histPD      = new TH1F("MVA_PDERS", "MVA_PDERS", nbin, 0, 1);
    if (Use["PDERSD"])        histPDD     = new TH1F("MVA_PDERSD", "MVA_PDERSD", nbin, 0, 1);
    if (Use["PDERSPCA"])      histPDPCA   = new TH1F("MVA_PDERSPCA", "MVA_PDERSPCA", nbin, 0, 1);
    if (Use["KNN"])           histKNN     = new TH1F("MVA_KNN", "MVA_KNN", nbin, 0, 1);
    if (Use["HMatrix"])       histHm      = new TH1F("MVA_HMatrix", "MVA_HMatrix", nbin, -0.95, 1.55);
    if (Use["Fisher"])        histFi      = new TH1F("MVA_Fisher", "MVA_Fisher", nbin, -4, 4);
    if (Use["FisherG"])       histFiG     = new TH1F("MVA_FisherG", "MVA_FisherG", nbin, -1, 1);
    if (Use["BoostedFisher"]) histFiB     = new TH1F("MVA_BoostedFisher", "MVA_BoostedFisher", nbin, -2, 2);
    if (Use["LD"])            histLD      = new TH1F("MVA_LD", "MVA_LD", nbin, -2, 2);
    if (Use["MLP"])           histNn      = new TH1F("MVA_MLP", "MVA_MLP", nbin, -1.25, 1.5);
    if (Use["MLPBFGS"])       histNnbfgs  = new TH1F("MVA_MLPBFGS", "MVA_MLPBFGS", nbin, -1.25, 1.5);
    if (Use["MLPBNN"])        histNnbnn   = new TH1F("MVA_MLPBNN", "MVA_MLPBNN", nbin, -1.25, 1.5);
    if (Use["CFMlpANN"])      histNnC     = new TH1F("MVA_CFMlpANN", "MVA_CFMlpANN", nbin, 0, 1);
    if (Use["TMlpANN"])       histNnT     = new TH1F("MVA_TMlpANN", "MVA_TMlpANN", nbin, -1.3, 1.3);
    if (Use["DNN_GPU"])       histDnnGpu  = new TH1F("MVA_DNN_GPU", "MVA_DNN_GPU", nbin, -0.1, 1.1);
    if (Use["DNN_CPU"])       histDnnCpu  = new TH1F("MVA_DNN_CPU", "MVA_DNN_CPU", nbin, -0.1, 1.1);
    if (Use["BDT"])           histBdt     = new TH1F("MVA_BDT", "MVA_BDT", nbin, -0.8, 0.8);
    if (Use["BDTG"])          histBdtG    = new TH1F("MVA_BDTG", "MVA_BDTG", nbin, -1.0, 1.0);
    if (Use["BDTB"])          histBdtB    = new TH1F("MVA_BDTB", "MVA_BDTB", nbin, -1.0, 1.0);
    if (Use["BDTD"])          histBdtD    = new TH1F("MVA_BDTD", "MVA_BDTD", nbin, -0.8, 0.8);
    if (Use["BDTF"])          histBdtF    = new TH1F("MVA_BDTF", "MVA_BDTF", nbin, -1.0, 1.0);
    if (Use["RuleFit"])       histRf      = new TH1F("MVA_RuleFit", "MVA_RuleFit", nbin, -2.0, 2.0);
    if (Use["SVM"])           histSVM     = new TH1F("MVA_SVM", "MVA_SVM", nbin, 0.0, 1.0);
    if (Use["SVM_Gauss"])     histSVMG    = new TH1F("MVA_SVM_Gauss", "MVA_SVM_Gauss", nbin, 0.0, 1.0);
    if (Use["SVM_Poly"])      histSVMP    = new TH1F("MVA_SVM_Poly", "MVA_SVM_Poly", nbin, 0.0, 1.0);
    if (Use["SVM_Lin"])       histSVML    = new TH1F("MVA_SVM_Lin", "MVA_SVM_Lin", nbin, 0.0, 1.0);
    if (Use["FDA_MT"])        histFDAMT   = new TH1F("MVA_FDA_MT", "MVA_FDA_MT", nbin, -2.0, 3.0);
    if (Use["FDA_GA"])        histFDAGA   = new TH1F("MVA_FDA_GA", "MVA_FDA_GA", nbin, -2.0, 3.0);
    if (Use["Category"])      histCat     = new TH1F("MVA_Category", "MVA_Category", nbin, -2., 2.);
    if (Use["Plugin"])        histPBdt    = new TH1F("MVA_PBDT", "MVA_BDT", nbin, -0.8, 0.8);

    // PDEFoam also returns per-event error, fill in histogram, and also fill significance
    if (Use["PDEFoam"]) 
    {
        histPDEFoam    = new TH1F( "MVA_PDEFoam",       "MVA_PDEFoam",              nbin,  0, 1 );
        histPDEFoamErr = new TH1F( "MVA_PDEFoamErr",    "MVA_PDEFoam error",        nbin,  0, 1 );
        histPDEFoamSig = new TH1F( "MVA_PDEFoamSig",    "MVA_PDEFoam significance", nbin,  0, 10 );
    }

    // Book example histogram for probability (the other methods are done similarly)
    TH1F *probHistFi(0), *rarityHistFi(0);
    if (Use["Fisher"]) 
    {
        probHistFi   = new TH1F( "MVA_Fisher_Proba",  "MVA_Fisher_Proba",  nbin, 0, 1 );
        rarityHistFi = new TH1F( "MVA_Fisher_Rarity", "MVA_Fisher_Rarity", nbin, 0, 1 );
    }

    int Nfile;
    const char *input_file[1000];
    char input_file_tem[100];
    char input_file_tem1[100];
    char input_file_tem2[100];
    char input_file_tem3[100];
    char input_file_tem4[100];
    char input_file_tem5[100];
    char input_file_tem6[100];
    char input_file_tem7[100];
    for(int i=0; i<100; i++)
    {
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

    for(int i=0;i<10;i++)
    {
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

    if(data_embed)
    {
        input_file[0] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_e-_embedHijing_50kHz_bkg_0_20fm-0000000004-00000_POSCOR_anaTutorial_50evt_20embed_e-.root";
        input_file[1] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_e-_embedHijing_50kHz_bkg_0_20fm-0000000004-00002_POSCOR_anaTutorial_50evt_20embed_e-.root";
        input_file[2] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_e-_embedHijing_50kHz_bkg_0_20fm-0000000004-00003_POSCOR_anaTutorial_50evt_20embed_e-.root";
    }
    if(data_single)
    {
        input_file[0] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_0_POSCOR.root_anaTutorial.root";  //without truthflavor
        input_file[1] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_01_POSCOR_anaTutorial.root";      //without truthflavor
        input_file[2] ="/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_02_POSCOR_anaTutorial.root";      //without truthflavor
    }

    float EOP=0.01;
    float HOM=0.01;

    //Nfile //test:169-188; 189-208
    for(int ifile=189;ifile<209;ifile++)
    {
        if(data_single & (ifile==366 or ifile==458 or ifile==450 or ifile==449)) continue;
        
        // why is 481?
        if(data_single & ifile>=481 && ifile<(481+72)) 
        {
            int ien=ifile-481;
            sprintf(input_file_tem7,"/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_0%d_anaTutorial.root",ien);
            input_file[ifile]=input_file_tem7;
        }
        if(data_embed & ifile>=89 && ifile<=168) continue;
 
        printf("file name is %s\n",input_file[ifile]);
 
 
        /////////////////////////
        TTree *readtree = nullptr;
        TFile *file4 = nullptr;
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
       
        for (Long64_t ievt=0; ievt<readtree->GetEntries();ievt++) 
        {
            // if (ievt%1000 == 0) cout << "--- ... Processing event: " <<readtree->GetEntries()<<"; "<< ievt << endl;
            // cout << "--- ... Processing event: " <<readtree->GetEntries()<<"; "<< ievt << endl;
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
 
            if(data_single && ifile<=12) gflavor2=11;
            if(data_single && !(ifile<=12 or ifile==113 or ifile==117 or ifile==121)) gflavor2=truthflavor;
            if(data_single && (ifile==113 or ifile==117 or ifile==121)) gflavor2=-2122;
 
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
 
            // flavor? pid?
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
 
             if(TMath::Abs(gflavor2)==11) 
             {
                h1EOP_e->Fill(EOP);
                h1HOM_e->Fill(HOM);
                h1CEMCchi2_e->Fill(cemcchi2);
                h1pt_cut->Fill(pt);
             }
             h1flavor_1->Fill(gflavor2);
 
            // h1EOP->Fill(EOP);
            // h1EcOP->Fill(EOP);
            // h1HOM->Fill(HOM);
            // h1CEMCchi2->Fill(cemcchi2);
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
 
            // Return the MVA outputs and fill into histogram
            if (Use["BDT"          ])   histBdt    ->Fill( reader->EvaluateMVA( "BDT method"           ) );
            if (Use["BDTG"         ])   histBdtG   ->Fill( reader->EvaluateMVA( "BDTG method"          ) );
            if (Use["BDTB"         ])   histBdtB   ->Fill( reader->EvaluateMVA( "BDTB method"          ) );
            if (Use["BDTD"         ])   histBdtD   ->Fill( reader->EvaluateMVA( "BDTD method"          ) );
            if (Use["BDTF"         ])   histBdtF   ->Fill( reader->EvaluateMVA( "BDTF method"          ) );

            
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
 
            if(TMath::Abs(gflavor2)==11 & var1>0.912 & var2<0.2) Nelectron_cuts=Nelectron_cuts+1; //traditional cuts: 3 vars：var1>0.908 &var2<0.2; 4 vars：var1>0.909 & var2<0.2; embed var1>0.912 & var2<0.2
 
            for(int i=0;i<5;i++)
            {
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
          }
       }//ievt
       file4->Close();
   }//ifile

}