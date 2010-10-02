
/*
 * ALMA - Atacama Large Millimeter Array
 * (c) European Southern Observatory, 2002
 * (c) Associated Universities Inc., 2002
 * Copyright by ESO (in the framework of the ALMA collaboration),
 * Copyright by AUI (in the framework of the ALMA collaboration),
 * All rights reserved.
 * 
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY, without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111-1307  USA
 * 
 * /////////////////////////////////////////////////////////////////
 * // WARNING!  DO NOT MODIFY THIS FILE!                          //
 * //  ---------------------------------------------------------  //
 * // | This is generated code!  Do not modify this file.       | //
 * // | Any changes will be lost when the file is re-generated. | //
 * //  ---------------------------------------------------------  //
 * /////////////////////////////////////////////////////////////////
 *
 * File CReceiverBand.cpp
 */
#include <sstream>
#include <CReceiverBand.h>
#include <string>
using namespace std;

	
const std::string& CReceiverBand::sALMA_RB_01 = "ALMA_RB_01";
	
const std::string& CReceiverBand::sALMA_RB_02 = "ALMA_RB_02";
	
const std::string& CReceiverBand::sALMA_RB_03 = "ALMA_RB_03";
	
const std::string& CReceiverBand::sALMA_RB_04 = "ALMA_RB_04";
	
const std::string& CReceiverBand::sALMA_RB_05 = "ALMA_RB_05";
	
const std::string& CReceiverBand::sALMA_RB_06 = "ALMA_RB_06";
	
const std::string& CReceiverBand::sALMA_RB_07 = "ALMA_RB_07";
	
const std::string& CReceiverBand::sALMA_RB_08 = "ALMA_RB_08";
	
const std::string& CReceiverBand::sALMA_RB_09 = "ALMA_RB_09";
	
const std::string& CReceiverBand::sALMA_RB_10 = "ALMA_RB_10";
	
const std::string& CReceiverBand::sALMA_HOLOGRAPHY_RECEIVER = "ALMA_HOLOGRAPHY_RECEIVER";
	
const std::string& CReceiverBand::sBURE_01 = "BURE_01";
	
const std::string& CReceiverBand::sBURE_02 = "BURE_02";
	
const std::string& CReceiverBand::sBURE_03 = "BURE_03";
	
const std::string& CReceiverBand::sBURE_04 = "BURE_04";
	
const std::string& CReceiverBand::sUNSPECIFIED = "UNSPECIFIED";
	
const std::vector<std::string> CReceiverBand::sReceiverBandSet() {
    std::vector<std::string> enumSet;
    
    enumSet.insert(enumSet.end(), CReceiverBand::sALMA_RB_01);
    
    enumSet.insert(enumSet.end(), CReceiverBand::sALMA_RB_02);
    
    enumSet.insert(enumSet.end(), CReceiverBand::sALMA_RB_03);
    
    enumSet.insert(enumSet.end(), CReceiverBand::sALMA_RB_04);
    
    enumSet.insert(enumSet.end(), CReceiverBand::sALMA_RB_05);
    
    enumSet.insert(enumSet.end(), CReceiverBand::sALMA_RB_06);
    
    enumSet.insert(enumSet.end(), CReceiverBand::sALMA_RB_07);
    
    enumSet.insert(enumSet.end(), CReceiverBand::sALMA_RB_08);
    
    enumSet.insert(enumSet.end(), CReceiverBand::sALMA_RB_09);
    
    enumSet.insert(enumSet.end(), CReceiverBand::sALMA_RB_10);
    
    enumSet.insert(enumSet.end(), CReceiverBand::sALMA_HOLOGRAPHY_RECEIVER);
    
    enumSet.insert(enumSet.end(), CReceiverBand::sBURE_01);
    
    enumSet.insert(enumSet.end(), CReceiverBand::sBURE_02);
    
    enumSet.insert(enumSet.end(), CReceiverBand::sBURE_03);
    
    enumSet.insert(enumSet.end(), CReceiverBand::sBURE_04);
    
    enumSet.insert(enumSet.end(), CReceiverBand::sUNSPECIFIED);
        
    return enumSet;
}

	

	
	
const std::string& CReceiverBand::hALMA_RB_01 = "ALMA Receiver band 01";
	
const std::string& CReceiverBand::hALMA_RB_02 = "ALMA Receiver band 02";
	
const std::string& CReceiverBand::hALMA_RB_03 = "ALMA Receiver band 03";
	
const std::string& CReceiverBand::hALMA_RB_04 = "ALMA Receiver band 04";
	
const std::string& CReceiverBand::hALMA_RB_05 = "ALMA Receiver band 05";
	
const std::string& CReceiverBand::hALMA_RB_06 = "ALMA Receiver band 06";
	
const std::string& CReceiverBand::hALMA_RB_07 = "ALMA Receiver band 07";
	
const std::string& CReceiverBand::hALMA_RB_08 = "ALMA Receiver band 08";
	
const std::string& CReceiverBand::hALMA_RB_09 = "ALMA Receiver band 09";
	
const std::string& CReceiverBand::hALMA_RB_10 = "ALMA Receiver band 10";
	
const std::string& CReceiverBand::hALMA_HOLOGRAPHY_RECEIVER = "Alma transmitter Holography receiver.";
	
const std::string& CReceiverBand::hBURE_01 = "Plateau de Bure receiver band #1.";
	
const std::string& CReceiverBand::hBURE_02 = "Plateau de Bure receiver band #2.";
	
const std::string& CReceiverBand::hBURE_03 = "Plateau de Bure receiver band #3.";
	
const std::string& CReceiverBand::hBURE_04 = "Plateau de Bure receiver band #4";
	
const std::string& CReceiverBand::hUNSPECIFIED = "receiver band of unspecified origin.";
	
const std::vector<std::string> CReceiverBand::hReceiverBandSet() {
    std::vector<std::string> enumSet;
    
    enumSet.insert(enumSet.end(), CReceiverBand::hALMA_RB_01);
    
    enumSet.insert(enumSet.end(), CReceiverBand::hALMA_RB_02);
    
    enumSet.insert(enumSet.end(), CReceiverBand::hALMA_RB_03);
    
    enumSet.insert(enumSet.end(), CReceiverBand::hALMA_RB_04);
    
    enumSet.insert(enumSet.end(), CReceiverBand::hALMA_RB_05);
    
    enumSet.insert(enumSet.end(), CReceiverBand::hALMA_RB_06);
    
    enumSet.insert(enumSet.end(), CReceiverBand::hALMA_RB_07);
    
    enumSet.insert(enumSet.end(), CReceiverBand::hALMA_RB_08);
    
    enumSet.insert(enumSet.end(), CReceiverBand::hALMA_RB_09);
    
    enumSet.insert(enumSet.end(), CReceiverBand::hALMA_RB_10);
    
    enumSet.insert(enumSet.end(), CReceiverBand::hALMA_HOLOGRAPHY_RECEIVER);
    
    enumSet.insert(enumSet.end(), CReceiverBand::hBURE_01);
    
    enumSet.insert(enumSet.end(), CReceiverBand::hBURE_02);
    
    enumSet.insert(enumSet.end(), CReceiverBand::hBURE_03);
    
    enumSet.insert(enumSet.end(), CReceiverBand::hBURE_04);
    
    enumSet.insert(enumSet.end(), CReceiverBand::hUNSPECIFIED);
        
    return enumSet;
}
   	

std::string CReceiverBand::name(const ReceiverBandMod::ReceiverBand& f) {
    switch (f) {
    
    case ReceiverBandMod::ALMA_RB_01:
      return CReceiverBand::sALMA_RB_01;
    
    case ReceiverBandMod::ALMA_RB_02:
      return CReceiverBand::sALMA_RB_02;
    
    case ReceiverBandMod::ALMA_RB_03:
      return CReceiverBand::sALMA_RB_03;
    
    case ReceiverBandMod::ALMA_RB_04:
      return CReceiverBand::sALMA_RB_04;
    
    case ReceiverBandMod::ALMA_RB_05:
      return CReceiverBand::sALMA_RB_05;
    
    case ReceiverBandMod::ALMA_RB_06:
      return CReceiverBand::sALMA_RB_06;
    
    case ReceiverBandMod::ALMA_RB_07:
      return CReceiverBand::sALMA_RB_07;
    
    case ReceiverBandMod::ALMA_RB_08:
      return CReceiverBand::sALMA_RB_08;
    
    case ReceiverBandMod::ALMA_RB_09:
      return CReceiverBand::sALMA_RB_09;
    
    case ReceiverBandMod::ALMA_RB_10:
      return CReceiverBand::sALMA_RB_10;
    
    case ReceiverBandMod::ALMA_HOLOGRAPHY_RECEIVER:
      return CReceiverBand::sALMA_HOLOGRAPHY_RECEIVER;
    
    case ReceiverBandMod::BURE_01:
      return CReceiverBand::sBURE_01;
    
    case ReceiverBandMod::BURE_02:
      return CReceiverBand::sBURE_02;
    
    case ReceiverBandMod::BURE_03:
      return CReceiverBand::sBURE_03;
    
    case ReceiverBandMod::BURE_04:
      return CReceiverBand::sBURE_04;
    
    case ReceiverBandMod::UNSPECIFIED:
      return CReceiverBand::sUNSPECIFIED;
    	
    }
    return std::string("");
}

	

	
std::string CReceiverBand::help(const ReceiverBandMod::ReceiverBand& f) {
    switch (f) {
    
    case ReceiverBandMod::ALMA_RB_01:
      return CReceiverBand::hALMA_RB_01;
    
    case ReceiverBandMod::ALMA_RB_02:
      return CReceiverBand::hALMA_RB_02;
    
    case ReceiverBandMod::ALMA_RB_03:
      return CReceiverBand::hALMA_RB_03;
    
    case ReceiverBandMod::ALMA_RB_04:
      return CReceiverBand::hALMA_RB_04;
    
    case ReceiverBandMod::ALMA_RB_05:
      return CReceiverBand::hALMA_RB_05;
    
    case ReceiverBandMod::ALMA_RB_06:
      return CReceiverBand::hALMA_RB_06;
    
    case ReceiverBandMod::ALMA_RB_07:
      return CReceiverBand::hALMA_RB_07;
    
    case ReceiverBandMod::ALMA_RB_08:
      return CReceiverBand::hALMA_RB_08;
    
    case ReceiverBandMod::ALMA_RB_09:
      return CReceiverBand::hALMA_RB_09;
    
    case ReceiverBandMod::ALMA_RB_10:
      return CReceiverBand::hALMA_RB_10;
    
    case ReceiverBandMod::ALMA_HOLOGRAPHY_RECEIVER:
      return CReceiverBand::hALMA_HOLOGRAPHY_RECEIVER;
    
    case ReceiverBandMod::BURE_01:
      return CReceiverBand::hBURE_01;
    
    case ReceiverBandMod::BURE_02:
      return CReceiverBand::hBURE_02;
    
    case ReceiverBandMod::BURE_03:
      return CReceiverBand::hBURE_03;
    
    case ReceiverBandMod::BURE_04:
      return CReceiverBand::hBURE_04;
    
    case ReceiverBandMod::UNSPECIFIED:
      return CReceiverBand::hUNSPECIFIED;
    	
    }
    return std::string("");
}
   	

ReceiverBandMod::ReceiverBand CReceiverBand::newReceiverBand(const std::string& name) {
		
    if (name == CReceiverBand::sALMA_RB_01) {
        return ReceiverBandMod::ALMA_RB_01;
    }
    	
    if (name == CReceiverBand::sALMA_RB_02) {
        return ReceiverBandMod::ALMA_RB_02;
    }
    	
    if (name == CReceiverBand::sALMA_RB_03) {
        return ReceiverBandMod::ALMA_RB_03;
    }
    	
    if (name == CReceiverBand::sALMA_RB_04) {
        return ReceiverBandMod::ALMA_RB_04;
    }
    	
    if (name == CReceiverBand::sALMA_RB_05) {
        return ReceiverBandMod::ALMA_RB_05;
    }
    	
    if (name == CReceiverBand::sALMA_RB_06) {
        return ReceiverBandMod::ALMA_RB_06;
    }
    	
    if (name == CReceiverBand::sALMA_RB_07) {
        return ReceiverBandMod::ALMA_RB_07;
    }
    	
    if (name == CReceiverBand::sALMA_RB_08) {
        return ReceiverBandMod::ALMA_RB_08;
    }
    	
    if (name == CReceiverBand::sALMA_RB_09) {
        return ReceiverBandMod::ALMA_RB_09;
    }
    	
    if (name == CReceiverBand::sALMA_RB_10) {
        return ReceiverBandMod::ALMA_RB_10;
    }
    	
    if (name == CReceiverBand::sALMA_HOLOGRAPHY_RECEIVER) {
        return ReceiverBandMod::ALMA_HOLOGRAPHY_RECEIVER;
    }
    	
    if (name == CReceiverBand::sBURE_01) {
        return ReceiverBandMod::BURE_01;
    }
    	
    if (name == CReceiverBand::sBURE_02) {
        return ReceiverBandMod::BURE_02;
    }
    	
    if (name == CReceiverBand::sBURE_03) {
        return ReceiverBandMod::BURE_03;
    }
    	
    if (name == CReceiverBand::sBURE_04) {
        return ReceiverBandMod::BURE_04;
    }
    	
    if (name == CReceiverBand::sUNSPECIFIED) {
        return ReceiverBandMod::UNSPECIFIED;
    }
    
    throw badString(name);
}

ReceiverBandMod::ReceiverBand CReceiverBand::literal(const std::string& name) {
		
    if (name == CReceiverBand::sALMA_RB_01) {
        return ReceiverBandMod::ALMA_RB_01;
    }
    	
    if (name == CReceiverBand::sALMA_RB_02) {
        return ReceiverBandMod::ALMA_RB_02;
    }
    	
    if (name == CReceiverBand::sALMA_RB_03) {
        return ReceiverBandMod::ALMA_RB_03;
    }
    	
    if (name == CReceiverBand::sALMA_RB_04) {
        return ReceiverBandMod::ALMA_RB_04;
    }
    	
    if (name == CReceiverBand::sALMA_RB_05) {
        return ReceiverBandMod::ALMA_RB_05;
    }
    	
    if (name == CReceiverBand::sALMA_RB_06) {
        return ReceiverBandMod::ALMA_RB_06;
    }
    	
    if (name == CReceiverBand::sALMA_RB_07) {
        return ReceiverBandMod::ALMA_RB_07;
    }
    	
    if (name == CReceiverBand::sALMA_RB_08) {
        return ReceiverBandMod::ALMA_RB_08;
    }
    	
    if (name == CReceiverBand::sALMA_RB_09) {
        return ReceiverBandMod::ALMA_RB_09;
    }
    	
    if (name == CReceiverBand::sALMA_RB_10) {
        return ReceiverBandMod::ALMA_RB_10;
    }
    	
    if (name == CReceiverBand::sALMA_HOLOGRAPHY_RECEIVER) {
        return ReceiverBandMod::ALMA_HOLOGRAPHY_RECEIVER;
    }
    	
    if (name == CReceiverBand::sBURE_01) {
        return ReceiverBandMod::BURE_01;
    }
    	
    if (name == CReceiverBand::sBURE_02) {
        return ReceiverBandMod::BURE_02;
    }
    	
    if (name == CReceiverBand::sBURE_03) {
        return ReceiverBandMod::BURE_03;
    }
    	
    if (name == CReceiverBand::sBURE_04) {
        return ReceiverBandMod::BURE_04;
    }
    	
    if (name == CReceiverBand::sUNSPECIFIED) {
        return ReceiverBandMod::UNSPECIFIED;
    }
    
    throw badString(name);
}

ReceiverBandMod::ReceiverBand CReceiverBand::from_int(unsigned int i) {
	vector<string> names = sReceiverBandSet();
	if (i >= names.size()) throw badInt(i);
	return newReceiverBand(names.at(i));
}

	

string CReceiverBand::badString(const string& name) {
	return "'"+name+"' does not correspond to any literal in the enumeration 'ReceiverBand'.";
}

string CReceiverBand::badInt(unsigned int i) {
	ostringstream oss ;
	oss << "'" << i << "' is out of range for the enumeration 'ReceiverBand'.";
	return oss.str();
}

