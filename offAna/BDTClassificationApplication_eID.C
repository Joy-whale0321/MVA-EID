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

void BDTClassificationApplication_eID( TString myMethodList = "" )
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

    // Book the MVA methods
    bool W_all=true;
    bool W_all_ecore=false;
    bool W_allN=false;
    bool W_antiproton=false;
    bool W_pion=false;
    bool W_Kion=false;

    bool data_single=true;
    bool data_embed=false;

    TString dir;
    // if(W_all) dir = "/mnt/f/sPHSimu/sPHENIX/MVA/TMVAClassification1/dataset_allN_cutpt2/weights/";
    // if(W_all) dir = "dataset_allN_cutpt2_12/weights/";
    if(W_all) dir = "dataset_allN_cutpt6_12_embed/weights/";
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

            cout<<"weightfile is: "<<weightfile<<endl;
        }
    }

    UInt_t nbin = 100;
    TH1F *histBdt  = nullptr;
    TH1F *histBdtG = nullptr;
    TH1F *histBdtB = nullptr;
    TH1F *histBdtD = nullptr;
    TH1F *histBdtF = nullptr;
    if (Use["BDT"])     histBdt     = new TH1F("MVA_BDT", "MVA_BDT", nbin, -0.8, 0.8);
    if (Use["BDTG"])    histBdtG    = new TH1F("MVA_BDTG", "MVA_BDTG", nbin, -1.0, 1.0);
    if (Use["BDTB"])    histBdtB    = new TH1F("MVA_BDTB", "MVA_BDTB", nbin, -1.0, 1.0);
    if (Use["BDTD"])    histBdtD    = new TH1F("MVA_BDTD", "MVA_BDTD", nbin, -0.8, 0.8);
    if (Use["BDTF"])    histBdtF    = new TH1F("MVA_BDTF", "MVA_BDTF", nbin, -1.0, 1.0);

    // --- 与 E/p, Ecore/p, HOM, cemc_chi2 相关的直方图 ---
    TH1F *h1EOP = nullptr;    // E3x3/p
    TH1F *h1EOP_e = nullptr;  // E3x3/p（用于电子）
    TH1F *h1EOP_cut = nullptr; // E3x3/p（切后）
    TH1F *h1EcOP = nullptr;   // Ecore/p
    h1EOP      = new TH1F("h1EOP", "h1EOP", nbin, 0.0, 5.0);
    h1EOP_e    = new TH1F("h1EOP_e", "h1EOP_e", 50, 0.0, 2.0);
    h1EOP_cut  = new TH1F("h1EOP_cut", "h1EOP_cut", nbin, 0.0, 5.0);
    h1EcOP     = new TH1F("h1EcOP", "h1EcOP", nbin, 0.0, 5.0);

    // --- 与 HOM 与 cemc_chi2 相关的直方图 ---
    TH1F *h1HOM = nullptr;    // inH3x3/E3x3
    TH1F *h1HOM_e = nullptr;  // inH3x3/E3x3（用于电子）
    TH1F *h1CEMCchi2 = nullptr;    // CEMC cluster Chi2
    TH1F *h1CEMCchi2_e = nullptr;  // CEMC cluster Chi2（用于电子）
    h1HOM       = new TH1F("h1HOM", "h1HOM", nbin, 0.0, 5.0);
    h1HOM_e     = new TH1F("h1HOM_e", "h1HOM_e", nbin, 0.0, 5.0);
    h1CEMCchi2  = new TH1F("h1CEMCchi2", "h1CEMCchi2", nbin, 0.0, 20.0);
    h1CEMCchi2_e= new TH1F("h1CEMCchi2_e", "h1CEMCchi2_e", nbin, 0.0, 20.0);

    // --- 与动量相关的直方图 ---
    TH1F *h1pt = nullptr;
    TH1F *h1pt_cut = nullptr;
    h1pt      = new TH1F("h1pt", "h1pt", nbin, 0.0, 20.0);
    h1pt_cut  = new TH1F("h1pt_cut", "h1pt_cut", nbin, 0.0, 20.0);

    // --- 与粒子 flavor 相关的直方图 ---
    TH1F *h1flavor_1 = nullptr;
    TH1F *h1flavor_2 = nullptr;
    h1flavor_1 = new TH1F("h1flavor_1", "h1flavor_1", 3000, -3000.0, 3000.0);
    h1flavor_2 = new TH1F("h1flavor_2", "h1flavor_2", 3000, -3000.0, 3000.0);

    // --- 与 BDT 相关的直方图 ---
    TH1F *h1electron_BDT = nullptr;
    TH1F *h1Sall_BDT = nullptr;
    TH1F *h1background_BDT = nullptr;
    TH1F *h1background_pion_BDT = nullptr;
    TH1F *h1background_antiproton_BDT = nullptr;
    TH1F *h1background_all_BDT = nullptr;
    h1electron_BDT        = new TH1F("h1electron_BDT", "h1electron_BDT", nbin, -1.0, 1.0);
    h1Sall_BDT            = new TH1F("h1Sall_BDT", "h1Sall_BDT", nbin, -1.0, 1.0);
    h1background_BDT      = new TH1F("h1background_BDT", "h1background_BDT", nbin, -1.0, 1.0);
    h1background_pion_BDT = new TH1F("h1background_pion_BDT", "h1background_pion_BDT", nbin, -1.0, 1.0);
    h1background_antiproton_BDT = new TH1F("h1background_antiproton_BDT", "h1background_antiproton_BDT", nbin, -1.0, 1.0);
    h1background_all_BDT  = new TH1F("h1background_all_BDT", "h1background_all_BDT", nbin, -1.0, 1.0);

    // --- 与变量分布相关的直方图（用于 MVA 输入变量检查等）---
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

    // 定义并初始化与粒子动量和能量相关的直方图
    TH1F *h1_p_1 = nullptr;
    TH1F *h1_pt_1 = nullptr;
    TH1F *h1_Eemcal3x3_1 = nullptr;
    TH1F *h1_p_2 = nullptr;
    TH1F *h1_pt_2 = nullptr;
    TH1F *h1_Eemcal3x3_2 = nullptr;

    h1_p_1         = new TH1F("h1_p_1", "h1_p_1", 100, 1.5, 49.5);
    h1_pt_1        = new TH1F("h1_pt_1", "h1_pt_1", 100, 1.5, 29.5);
    h1_Eemcal3x3_1 = new TH1F("h1_Eemcal3x3_1", "h1_Eemcal3x3_1", 180, 1.5, 19.5);

    h1_p_2         = new TH1F("h1_p_2", "h1_p_2", 100, 1.5, 49.5);
    h1_pt_2        = new TH1F("h1_pt_2", "h1_pt_2", 100, 1.5, 29.5);
    h1_Eemcal3x3_2 = new TH1F("h1_Eemcal3x3_2", "h1_Eemcal3x3_2", 180, 1.5, 19.5);

    // 定义并初始化二维直方图，用于绘制MVA输出与其他变量（如pt、EOP、HOM、cemcchi2）的关系
    TH2F *h2_reponse_pt = nullptr;
    TH2F *h2_reponse_EOP = nullptr;
    TH2F *h2_reponse_HOM = nullptr;
    TH2F *h2_reponse_chi2 = nullptr;

    h2_reponse_pt    = new TH2F("h2_reponse_pt", "h2_reponse_pt", 50, -0.5, 0.5, 100, 1.5, 12.5);
    h2_reponse_EOP   = new TH2F("h2_reponse_EOP", "h2_reponse_EOP", 50, -0.5, 0.5, 40, 0.0, 4.0);
    h2_reponse_HOM   = new TH2F("h2_reponse_HOM", "h2_reponse_HOM", 50, -0.5, 0.5, 100, 0.0, 1.0);
    h2_reponse_chi2  = new TH2F("h2_reponse_chi2", "h2_reponse_chi2", 50, -0.5, 0.5, 200, 0.0, 20.0);

    // Efficiency calculator for cut method
    Int_t    nSelCutsGA = 0;
    Double_t effS       = 0.8;

    // counts for efficiency calculation
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

    // input file name setting 
    const char *input_file[1000];
    char input_file_tem7[100];
    for(int i=0; i<100; i++)
    {
        sprintf(input_file_tem7,"inatialization_%d",i);
    }

    if(data_embed)
    {
        input_file[0] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_e-_embedHijing_50kHz_bkg_0_20fm-0000000004-00000_POSCOR_anaTutorial_50evt_20embed_e-.root";
        input_file[1] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_e-_embedHijing_50kHz_bkg_0_20fm-0000000004-00002_POSCOR_anaTutorial_50evt_20embed_e-.root";
        input_file[2] ="/mnt/f/sPHSimu/sPHENIX/embed_data/G4sPHENIX_e-_embedHijing_50kHz_bkg_0_20fm-0000000004-00003_POSCOR_anaTutorial_50evt_20embed_e-.root";
    }
    if(data_single)
    {
        input_file[0] ="/mnt/d/cundian_data/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_0_POSCOR.root_anaTutorial.root";  //without truthflavor
        input_file[1] ="/mnt/d/cundian_data/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_01_POSCOR_anaTutorial.root";      //without truthflavor
        input_file[2] ="/mnt/d/cundian_data/single_data/G4sPHENIX_e-_eta0-1.1_0-20GeV_100_02_POSCOR_anaTutorial.root";      //without truthflavor
    }
    
    float EOP=0.01;
    float HOM=0.01;

    TStopwatch sw;
    sw.Start();

    // file loop
    for(int ifile=0;ifile<3;ifile++)
    {
        // if(data_single && (ifile==366 or ifile==458 or ifile==450 or ifile==449)) continue;
        
        // // file name setting
        // if(data_single && ifile>=481 && ifile<(481+72)) 
        // {
        //     int ien=ifile-481;
        //     sprintf(input_file_tem7,"/mnt/f/sPHSimu/sPHENIX/single_data/G4sPHENIX_e-_eta0-1.1_2-12GeV_400_0%d_anaTutorial.root",ien);
        //     input_file[ifile]=input_file_tem7;
        // }
        // if(data_embed && ifile>=89 && ifile<=168) continue;

        printf("file name is %s\n",input_file[ifile]);
        

        int nmvtx,nintt,ntpc,charge;
        float quality;
        double trpx,trpy,trpz,trpt,trp,m_tr_eta,trphi,trdca;
        double cemcdphi,cemcdeta,m_cemce3x3,cemce5x5,cemce,cemcecore,cemc_prob,cemc_chi2;
        double hcalindphi,hcalindeta,hcaline3x3,hcaline5x5,hcaline;
        double gflavor2,bimp2;
 
        int nmvtx,nintt,ntpc,m_charge;
        float quality;
        double m_tr_px,m_tr_py,m_tr_pz,m_tr_pt,m_tr_p,m_tr_eta,m_tr_phi,m_tr_dca;
        double m_cemcdphi,m_cemcdeta,m_cemce3x3,m_cemce5x5,m_cemce,cemc_ecore,cemc_prob,cemc_chi2;
        double m_hcalindphi,m_hcalindeta,m_hcaline3x3,m_hcaline5x5,m_hcaline;
        double truthflavor,bimp;

        TTree *readtree = nullptr;
        TFile *file4 = nullptr;
        file4 = TFile::Open(input_file[ifile]);
        readtree = (TTree*)file4->Get("tracktree");

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
        if(data_single && !(ifile<=12 or ifile==113 or ifile==117 or ifile==121 )) readtree->SetBranchAddress( "truthflavor", &truthflavor);
        // readtree->SetBranchAddress( "bimp", &bimp);
        bimp = 2;

        for (Long64_t ievt=0; ievt<readtree->GetEntries();ievt++) 
        {
            readtree->GetEntry(ievt);     

            if(data_embed) gflavor2=truthflavor;
 
            if(data_single && ifile<=12) gflavor2=11;
            if(data_single && !(ifile<=12 or ifile==113 or ifile==117 or ifile==121 )) gflavor2=truthflavor;
            if(data_single && (ifile==113 or ifile==117 or ifile==121)) gflavor2=-2122;

            bimp2=bimp;
            cout<<ifile<<"; "<<gflavor2<<"; "<<bimp2<<endl;
 
            float p2   = m_tr_p;
            float EOP  = m_cemce3x3/p2; // E3x3/p
            float EcOP = cemc_ecore/p2;  // Ecore/p
            float HOM  = m_hcaline3x3/m_cemce3x3; // EHcalin/EEmcal
            float dR   = TMath::Sqrt(m_cemcdphi*m_cemcdphi + m_cemcdeta*m_cemcdeta);
            float pt   = m_tr_pt;

            h1pt->Fill(pt);
            //h1EOP->Fill(EOP);
            h1EcOP->Fill(EOP); // ??? EcOP?
            h1HOM->Fill(HOM);
            h1CEMCchi2->Fill(cemc_chi2);
 
            // flavor? pid?
            if(gflavor2==11) N_raw=N_raw+1;
 
            if(gflavor2==11 && EOP>0.0 && EOP<20.0 && HOM>0.0 && HOM<20.0 && nmvtx>0 && nintt>0 && ntpc>20 && quality<10) 
            {
                N_track=N_track+1;
            }
 
            if(EOP>0.0 && EOP<20.0 && HOM>0.0 && HOM<20.0 && nmvtx>0 && nintt>0 && ntpc>20 && quality<10 && pt>2.0 && pt<=12.0) 
            {
                h1EOP->Fill(EOP);
            }
            
            if((gflavor2==11 or gflavor2==-2212 or gflavor2==-211 or gflavor2==-321) && nmvtx>0 && nintt>0 && quality<10 && (TMath::Abs(m_tr_eta)<=1.1) && EOP>0.0 && EOP<20.0 && HOM>0.0 && HOM<20.0 && pt>2.0 && pt<=13.0 && ntpc>20 && ntpc<=48 && cemc_prob>0.0 && cemc_prob<=1.0 && cemc_chi2>0.0 && cemc_chi2<20.0) 
            {
                if(gflavor2==11) N_track_pt2=N_track_pt2+1;
 
                if(TMath::Abs(gflavor2)==11) 
                {
                    h1EOP_e->Fill(EOP);
                    h1HOM_e->Fill(HOM);
                    h1CEMCchi2_e->Fill(cemc_chi2);
                    h1pt_cut->Fill(pt);
                }
                h1flavor_1->Fill(gflavor2);

                var1 = EOP;
                var2 = HOM;
                var3 = cemc_chi2;

                if (Use["BDT"          ])   histBdt    ->Fill( reader->EvaluateMVA( "BDT method"           ) );
                if (Use["BDTG"         ])   histBdtG   ->Fill( reader->EvaluateMVA( "BDTG method"          ) );
                if (Use["BDTB"         ])   histBdtB   ->Fill( reader->EvaluateMVA( "BDTB method"          ) );
                if (Use["BDTD"         ])   histBdtD   ->Fill( reader->EvaluateMVA( "BDTD method"          ) );
                if (Use["BDTF"         ])   histBdtF   ->Fill( reader->EvaluateMVA( "BDTF method"          ) );
            
                if(TMath::Abs(gflavor2)==11) NSall=NSall+1;
                if(TMath::Abs(gflavor2)==11) Nelectron=Nelectron+1;
                if(TMath::Abs(gflavor2)==211) Npion=Npion+1;
                if(TMath::Abs(gflavor2)==2212) Nantiproton=Nantiproton+1;
                if(TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) Nall=Nall+1;
                
                if(TMath::Abs(gflavor2)==11 && var1>0.912 && var2<0.2) Nelectron_cuts=Nelectron_cuts+1; //traditional cuts: 3 vars：var1>0.908 & var2<0.2; 4 vars：var1>0.909 & var2<0.2; embed var1>0.912 & var2<0.2
                
                // --------------------------------------------------------------------------------------------------
                if (Use["BDT"]) 
                {
                    float select=reader->EvaluateMVA("BDT method");
                    //std::cout <<"BDT select= " << select<< std::endl;
                    if(TMath::Abs(gflavor2)==11)  h1electron_BDT->Fill(select); // e 的 BDT eval
                    if(TMath::Abs(gflavor2)==11)  h1Sall_BDT->Fill(select); // e 的 BDT eval
                    if(TMath::Abs(gflavor2)==211) h1background_pion_BDT->Fill(select); // 211 的 BDT eval
                    if(TMath::Abs(gflavor2)==2212)h1background_antiproton_BDT->Fill(select); // 2212 的 BDT eval
                    if(TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) h1background_all_BDT->Fill(select); // 2212 211 321的 BDT eval
                    
                if(select>-0.39 && select<-0.35)
                    {
                        h1flavor_2->Fill(gflavor2);  
                        h1var1_EOP_2->Fill(var1);
                        h1var2_HOM_2->Fill(var2);
                        h1var3_Chi2_2->Fill(var3);
                        h1_p_2->Fill(p2);
                        h1_pt_2->Fill(pt);
                        h1_Eemcal3x3_2->Fill(m_cemce3x3);
                    }
                if(select>-0.49 && select<-0.43)
                    {
                        // h1flavor_1->Fill(gflavor2);
                        h1var1_EOP_1->Fill(var1);
                        h1var2_HOM_1->Fill(var2);
                        h1var3_Chi2_1->Fill(var3);
                        h1_p_1->Fill(p2);
                        h1_pt_1->Fill(pt);
                        h1_Eemcal3x3_1->Fill(m_cemce3x3);
                    }
       
                    if(TMath::Abs(gflavor2)==11)
                    {
                        h2_reponse_pt->Fill(select,pt);
                        h2_reponse_EOP->Fill(select,EOP);
                        h2_reponse_HOM->Fill(select,HOM);
                        h2_reponse_chi2->Fill(select,cemc_chi2);
                    }
       
                    if(TMath::Abs(gflavor2)==11 && select>0.1431) Nelectron_BDT=Nelectron_BDT+1;  //3 vars：select>0.1355; 4vars: select>0.138; embed: select>0.1431
       
                    for(int i=0;i<10;i++)
                    {
                        if(select>0.1431)
                        {
                            Npt[i]=2.0*i+2.0;
                            if((TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) && (pt>=Npt[i]-1.0) && pt<(Npt[i]+1.0)) //plot for all
                            // if((TMath::Abs(gflavor2)==211) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0))   //plot for pi-
                            // if((TMath::Abs(gflavor2)==321) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0))   //plot for K-
                            // if((TMath::Abs(gflavor2)==2212) & (pt>=Npt[i]-1.0) & pt<(Npt[i]+1.0))  //plot for antiprotpn
                            { 
                                nall_BDT_pt[i]=nall_BDT_pt[i]+1;
                            }
       
                        } 
                    }
                    
                    ///////////////////////////////////////////
                    for(int i=0;i<5;i++)
                    {
                        if(select>0.1431) // 90% eID efficency; select>0.1360 for e3x3 cutpt2; select>0.20 for ecore cutpt2; select>0.15 for e3x3; select>0.10 for ecore;  select>0.18 for e3x3 cutpt2 embed
                        {
                            Nbimp[i]=4.0*i+2.0;
                            if((TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) && (bimp>=Nbimp[i]-2.0) && bimp<(Nbimp[i]+2.0)) //plot for all 
                            // if((TMath::Abs(gflavor2)==211) & (bimp>=Nbimp[i]-1.0) & bimp<(Nbimp[i]+1.0))  //plot for pi-
                            // if((TMath::Abs(gflavor2)==321) & (bimp>=Nbimp[i]-1.0) & bimp<(Nbimp[i]+1.0))  //plot for K-
                            // if((TMath::Abs(gflavor2)==2212) & (bimp>=Nbimp[i]-1.0) & bimp<(Nbimp[i]+1.0)) //plot for antiprotpn
                            { 
                                nall_BDT_bimp[i]=nall_BDT_bimp[i]+1;
                            }
       
                        } 
                    }
       
                    for(int i=0;i<10;i++)
                    {
                        pt_point[i]=i*2.0+2.0;
                        if(pt>(pt_point[i]-1.0) && pt<(pt_point[i]+1.0) )
                        {
                            if(gflavor2==11) N_electron_pt_BDT[i]=N_electron_pt_BDT[i]+1;
                            if(select>0.1431)
                            {
                                if(gflavor2==11) NEID_electron_pt_BDT[i]=NEID_electron_pt_BDT[i]+1;               
                            }
                        }
                    }
                    
                    for(int i=0;i<7;i++)
                    {
                    if(W_antiproton && data_embed)Ncut_BDT[i]=i*0.1-0.3;//antiproton weight enmbed
                    if(W_antiproton && data_single)Ncut_BDT[i]=i*0.1-0.245;//antiproton weight single
                        if(W_pion)Ncut_BDT[i]=i*0.1-0.2; //pion weight
                    if(W_all && data_embed)Ncut_BDT[i]=i*0.07-0.18; //all weight
                    if(W_all && data_single)Ncut_BDT[i]=i*0.058-0.18; //all weight
                    if(W_all_ecore && data_single)Ncut_BDT[i]=i*0.064-0.20; //all weight most=0.41
                        if(select>Ncut_BDT[i])
                        {
                             //std::cout <<Ncut_BDT[i]<< "; BDT selected electrons" << std::endl;
                             if(TMath::Abs(gflavor2)==11) nelectron_BDT[i]=nelectron_BDT[i]+1;
                             if(TMath::Abs(gflavor2)==11) nSall_BDT[i]=nSall_BDT[i]+1;
                             if(TMath::Abs(gflavor2)==211) npion_BDT[i]=npion_BDT[i]+1;
                             if(TMath::Abs(gflavor2)==2212) nantiproton_BDT[i]=nantiproton_BDT[i]+1;
                             if(TMath::Abs(gflavor2)==2212 or TMath::Abs(gflavor2)==211 or TMath::Abs(gflavor2)==321) nall_BDT[i]=nall_BDT[i]+1;
                            // std::cout << "nelectron_BDT= "<<nelectron_BDT[i]<< std::endl;
                            // std::cout << "nantiproton_BDT= "<<nantiproton_BDT[i]<< std::endl;
                        }
                        else
                        {
                            //std::cout << "BDT selected background" << std::endl;
                            // if(gflavor2==-211) npion_BDT=npion_BDT+1;
                            // if(gflavor2==-2212) nantiproton_BDT=nantiproton_BDT+1;
                        }
                    }
                } // if USE BDT
        
            } // if gflavor
        } // event (entries)
    } // file

    // efficiency calculation ---------------------------------------------------------------------------------------------
    // 定义仅用于 BDT、SVM 和 DNN_CPU 的相关数组
    float efficiency_electron_BDT[10], efficiency_Sall_BDT[10], rejection_antiproton_BDT[10], rejection_pion_BDT[10], rejection_all_BDT[10];
    float err_efficiency_electron_BDT[10], err_efficiency_Sall_BDT[10], err_rejection_antiproton_BDT[10], err_rejection_pion_BDT[10], err_rejection_all_BDT[10];
    float SBratio_antiproton_BDT[10], SBratio_pion_BDT[10], SBratio_all_BDT[10];

    float rejection_all_BDT_pt[10], err_rejection_all_BDT_pt[10];
    float rejection_all_BDT_pt_inverse[10], err_rejection_all_BDT_pt_inverse[10];
    float rejection_all_BDT_bimp[10], err_rejection_all_BDT_bimp[10];
    float rejection_all_BDT_bimp_inverse[10], err_rejection_all_BDT_bimp_inverse[10];

    float aa_pt_N[10], err_aa_pt_N[10];
    float cc_pt_N_BDT[10], err_cc_pt_N_BDT[10];

    // -------------------------
    // 1. 初始化所有数组为0
    for (int i = 0; i < 10; i++) 
    {
        efficiency_electron_BDT[i] = 0.0;
        efficiency_Sall_BDT[i] = 0.0;
        rejection_antiproton_BDT[i] = 0.0;
        rejection_pion_BDT[i] = 0.0;
        rejection_all_BDT[i] = 0.0;

        err_efficiency_electron_BDT[i] = 0.0;
        err_efficiency_Sall_BDT[i] = 0.0;
        err_rejection_antiproton_BDT[i] = 0.0;
        err_rejection_pion_BDT[i] = 0.0;
        err_rejection_all_BDT[i] = 0.0;

        SBratio_antiproton_BDT[i] = 0.0;
        SBratio_pion_BDT[i] = 0.0;
        SBratio_all_BDT[i] = 0.0;

        rejection_all_BDT_pt[i] = 0.0;
        err_rejection_all_BDT_pt[i] = 0.0;

        rejection_all_BDT_pt_inverse[i] = 0.0;
        err_rejection_all_BDT_pt_inverse[i] = 0.0;

        rejection_all_BDT_bimp[i] = 0.0;
        err_rejection_all_BDT_bimp[i] = 0.0;

        rejection_all_BDT_bimp_inverse[i] = 0.0;
        err_rejection_all_BDT_bimp_inverse[i] = 0.0;

        aa_pt_N[i] = 0.0;
        err_aa_pt_N[i] = 0.0;
        cc_pt_N_BDT[i] = 0.0;
        err_cc_pt_N_BDT[i] = 0.0;
    }

    // -------------------------
    // 2. 计算 pt 分bin上的比值， BDT
    for (int i = 0; i < 10; i++) 
    {
        aa_pt_N[i] = pt_point[i];
        err_aa_pt_N[i] = 1.0;

        if (N_electron_pt_BDT[i] > 0 && NEID_electron_pt_BDT[i] > 0) 
        {
            cc_pt_N_BDT[i] = 1.0 * NEID_electron_pt_BDT[i] / N_electron_pt_BDT[i];
            err_cc_pt_N_BDT[i] = 1.0 * TMath::Sqrt((1.0 / NEID_electron_pt_BDT[i] + 1.0 / N_electron_pt_BDT[i])) * cc_pt_N_BDT[i];
        }
    }

    // -------------------------
    // 3. BDT 方法的统计计算（假定相关变量如 nelectron_BDT、nSall_BDT、nantiproton_BDT、npion_BDT、nall_BDT 均已定义）
    for (int i = 0; i < 7; i++) 
    {
        if (Nelectron > 0 && nelectron_BDT[i] > 0) 
        {
            efficiency_electron_BDT[i] = 1.0 * nelectron_BDT[i] / Nelectron;
            err_efficiency_electron_BDT[i] = 1.0 * TMath::Sqrt((1.0 / nelectron_BDT[i] + 1.0 / Nelectron)) * efficiency_electron_BDT[i];
        }

        if (NSall > 0 && nSall_BDT[i] > 0) 
        {
            efficiency_Sall_BDT[i] = 1.0 * nSall_BDT[i] / NSall;
            err_efficiency_Sall_BDT[i] = 1.0 * TMath::Sqrt((1.0 / nSall_BDT[i] + 1.0 / NSall)) * efficiency_Sall_BDT[i];
        }

        if (Nantiproton > 0 && nantiproton_BDT[i] > 0) 
        {
            rejection_antiproton_BDT[i] = 1.0 * Nantiproton / nantiproton_BDT[i];
            err_rejection_antiproton_BDT[i] = 1.0 * TMath::Sqrt((1.0 / Nantiproton + 1.0 / nantiproton_BDT[i])) * rejection_antiproton_BDT[i];
            SBratio_antiproton_BDT[i] = 1.0 * nelectron_BDT[i] / TMath::Sqrt(nantiproton_BDT[i] + nelectron_BDT[i]);
        }

        if (Npion > 0 && npion_BDT[i] > 0) 
        {
            rejection_pion_BDT[i] = 1.0 * Npion / npion_BDT[i];
            err_rejection_pion_BDT[i] = 1.0 * TMath::Sqrt((1.0 / Npion + 1.0 / npion_BDT[i])) * rejection_pion_BDT[i];
            SBratio_pion_BDT[i] = 1.0 * nelectron_BDT[i] / TMath::Sqrt(npion_BDT[i] + nelectron_BDT[i]);
        }

        if (Nall > 0 && nall_BDT[i] > 0) 
        {
            rejection_all_BDT[i] = 1.0 * Nall / nall_BDT[i];
            err_rejection_all_BDT[i] = 1.0 * TMath::Sqrt((1.0 / Nall + 1.0 / nall_BDT[i])) * rejection_all_BDT[i];
            SBratio_all_BDT[i] = 1.0 * nSall_BDT[i] / TMath::Sqrt(nall_BDT[i] + nSall_BDT[i]);
        }
    }

    // BDT 方法：pt 分箱的背景拒绝计算
    for (int i = 0; i < 9; i++) 
    {
        if (Nall_pt[i] > 0 && nall_BDT_pt[i] > 0) 
        {
            rejection_all_BDT_pt[i] = 1.0 * Nall_pt[i] / nall_BDT_pt[i];
            err_rejection_all_BDT_pt[i] = 1.0 * TMath::Sqrt((1.0 / Nall_pt[i] + 1.0 / nall_BDT_pt[i])) * rejection_all_BDT_pt[i];

            rejection_all_BDT_pt_inverse[i] = 1.0 * nall_BDT_pt[i] / Nall_pt[i];
            err_rejection_all_BDT_pt_inverse[i] = 1.0 * TMath::Sqrt((1.0 / Nall_pt[i] + 1.0 / nall_BDT_pt[i])) * rejection_all_BDT_pt_inverse[i];
        }
    }

    // BDT 方法：bimp 分箱的背景拒绝计算
    for (int i = 0; i < 5; i++)   
    {
        if (Nall_bimp[i] > 0 && nall_BDT_bimp[i] > 0) 
        {
            rejection_all_BDT_bimp[i] = 1.0 * Nall_bimp[i] / nall_BDT_bimp[i];
            err_rejection_all_BDT_bimp[i] = 1.0 * TMath::Sqrt((1.0 / Nall_bimp[i] + 1.0 / nall_BDT_bimp[i])) * rejection_all_BDT_bimp[i];

            rejection_all_BDT_bimp_inverse[i] = 1.0 * nall_BDT_bimp[i] / Nall_bimp[i];
            err_rejection_all_BDT_bimp_inverse[i] = 1.0 * TMath::Sqrt((1.0 / Nall_bimp[i] + 1.0 / nall_BDT_bimp[i])) * rejection_all_BDT_bimp_inverse[i];
        }
    }

    // Get elapsed time
    sw.Stop();
    std::cout << "--- End of event loop: "; sw.Print();

    // 创建输出 ROOT 文件，模式为 RECREATE（重写）
    TFile *outfile = TFile::Open("output.root", "RECREATE");

    // 将所有直方图写入文件（注意只写入已创建的对象）
    if(histBdt)           histBdt->Write();
    if(histBdtG)          histBdtG->Write();
    if(histBdtB)          histBdtB->Write();
    if(histBdtD)          histBdtD->Write();
    if(histBdtF)          histBdtF->Write();

    if(h1EOP)             h1EOP->Write();
    if(h1EOP_e)           h1EOP_e->Write();
    if(h1EOP_cut)         h1EOP_cut->Write();
    if(h1EcOP)            h1EcOP->Write();

    if(h1HOM)             h1HOM->Write();
    if(h1HOM_e)           h1HOM_e->Write();
    if(h1CEMCchi2)        h1CEMCchi2->Write();
    if(h1CEMCchi2_e)      h1CEMCchi2_e->Write();

    if(h1pt)              h1pt->Write();
    if(h1pt_cut)          h1pt_cut->Write();

    if(h1flavor_1)        h1flavor_1->Write();
    if(h1flavor_2)        h1flavor_2->Write();

    if(h1electron_BDT)    h1electron_BDT->Write();
    if(h1Sall_BDT)        h1Sall_BDT->Write();
    if(h1background_BDT)  h1background_BDT->Write();
    if(h1background_pion_BDT) h1background_pion_BDT->Write();
    if(h1background_antiproton_BDT) h1background_antiproton_BDT->Write();
    if(h1background_all_BDT) h1background_all_BDT->Write();

    if(h1var1_EOP_1)      h1var1_EOP_1->Write();
    if(h1var2_HOM_1)      h1var2_HOM_1->Write();
    if(h1var3_Chi2_1)     h1var3_Chi2_1->Write();
    if(h1var1_EOP_2)      h1var1_EOP_2->Write();
    if(h1var2_HOM_2)      h1var2_HOM_2->Write();
    if(h1var3_Chi2_2)     h1var3_Chi2_2->Write();

    if(h1_p_1)            h1_p_1->Write();
    if(h1_pt_1)           h1_pt_1->Write();
    if(h1_Eemcal3x3_1)    h1_Eemcal3x3_1->Write();
    if(h1_p_2)            h1_p_2->Write();
    if(h1_pt_2)           h1_pt_2->Write();
    if(h1_Eemcal3x3_2)    h1_Eemcal3x3_2->Write();

    if(h2_reponse_pt)     h2_reponse_pt->Write();
    if(h2_reponse_EOP)    h2_reponse_EOP->Write();
    if(h2_reponse_HOM)    h2_reponse_HOM->Write();
    if(h2_reponse_chi2)   h2_reponse_chi2->Write();

    // 写入文件并关闭
    outfile->Close();
    std::cout << "所有直方图已保存到 output.root" << std::endl;

} // BDTClassificationApplication_eID




void TMVAClassificationTraining_Test()
{
    std::cout <<"OK!!!!!!!!!!!!!!!!!!!!!"<< std::endl;
}

int main( int argc, char** argv )
{
    TString methodList;
    for (int i=1; i<argc; i++) 
    {
        TString regMethod(argv[i]);
        if(regMethod=="-b" || regMethod=="--batch") continue;
        if (!methodList.IsNull()) methodList += TString(",");
        methodList += regMethod;
    }
    BDTClassificationApplication_eID(methodList);
    TMVAClassificationTraining_Test();
    return 0;
}
