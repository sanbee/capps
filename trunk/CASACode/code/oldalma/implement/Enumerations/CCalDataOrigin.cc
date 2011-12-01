
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
 * File CCalDataOrigin.cpp
 */
#include <sstream>
#include <CCalDataOrigin.h>
#include <string>
using namespace std;

	
const std::string& CCalDataOrigin::sTOTAL_POWER = "TOTAL_POWER";
	
const std::string& CCalDataOrigin::sWVR = "WVR";
	
const std::string& CCalDataOrigin::sCHANNEL_AVERAGE_AUTO = "CHANNEL_AVERAGE_AUTO";
	
const std::string& CCalDataOrigin::sCHANNEL_AVERAGE_CROSS = "CHANNEL_AVERAGE_CROSS";
	
const std::string& CCalDataOrigin::sFULL_RESOLUTION_AUTO = "FULL_RESOLUTION_AUTO";
	
const std::string& CCalDataOrigin::sFULL_RESOLUTION_CROSS = "FULL_RESOLUTION_CROSS";
	
const std::string& CCalDataOrigin::sOPTICAL_POINTING = "OPTICAL_POINTING";
	
const std::string& CCalDataOrigin::sHOLOGRAPHY = "HOLOGRAPHY";
	
const std::vector<std::string> CCalDataOrigin::sCalDataOriginSet() {
    std::vector<std::string> enumSet;
    
    enumSet.insert(enumSet.end(), CCalDataOrigin::sTOTAL_POWER);
    
    enumSet.insert(enumSet.end(), CCalDataOrigin::sWVR);
    
    enumSet.insert(enumSet.end(), CCalDataOrigin::sCHANNEL_AVERAGE_AUTO);
    
    enumSet.insert(enumSet.end(), CCalDataOrigin::sCHANNEL_AVERAGE_CROSS);
    
    enumSet.insert(enumSet.end(), CCalDataOrigin::sFULL_RESOLUTION_AUTO);
    
    enumSet.insert(enumSet.end(), CCalDataOrigin::sFULL_RESOLUTION_CROSS);
    
    enumSet.insert(enumSet.end(), CCalDataOrigin::sOPTICAL_POINTING);
    
    enumSet.insert(enumSet.end(), CCalDataOrigin::sHOLOGRAPHY);
        
    return enumSet;
}

	

	
	
const std::string& CCalDataOrigin::hTOTAL_POWER = "Total Power data (from detectors)";
	
const std::string& CCalDataOrigin::hWVR = "Water vapour radiometrers";
	
const std::string& CCalDataOrigin::hCHANNEL_AVERAGE_AUTO = "Autocorrelations from channel average data";
	
const std::string& CCalDataOrigin::hCHANNEL_AVERAGE_CROSS = "Crosscorrelations from channel average data";
	
const std::string& CCalDataOrigin::hFULL_RESOLUTION_AUTO = "Autocorrelations from full-resolution data";
	
const std::string& CCalDataOrigin::hFULL_RESOLUTION_CROSS = "Cross correlations from full-resolution data";
	
const std::string& CCalDataOrigin::hOPTICAL_POINTING = "Optical pointing data";
	
const std::string& CCalDataOrigin::hHOLOGRAPHY = "data from holography receivers";
	
const std::vector<std::string> CCalDataOrigin::hCalDataOriginSet() {
    std::vector<std::string> enumSet;
    
    enumSet.insert(enumSet.end(), CCalDataOrigin::hTOTAL_POWER);
    
    enumSet.insert(enumSet.end(), CCalDataOrigin::hWVR);
    
    enumSet.insert(enumSet.end(), CCalDataOrigin::hCHANNEL_AVERAGE_AUTO);
    
    enumSet.insert(enumSet.end(), CCalDataOrigin::hCHANNEL_AVERAGE_CROSS);
    
    enumSet.insert(enumSet.end(), CCalDataOrigin::hFULL_RESOLUTION_AUTO);
    
    enumSet.insert(enumSet.end(), CCalDataOrigin::hFULL_RESOLUTION_CROSS);
    
    enumSet.insert(enumSet.end(), CCalDataOrigin::hOPTICAL_POINTING);
    
    enumSet.insert(enumSet.end(), CCalDataOrigin::hHOLOGRAPHY);
        
    return enumSet;
}
   	

std::string CCalDataOrigin::name(const CalDataOriginMod::CalDataOrigin& f) {
    switch (f) {
    
    case CalDataOriginMod::TOTAL_POWER:
      return CCalDataOrigin::sTOTAL_POWER;
    
    case CalDataOriginMod::WVR:
      return CCalDataOrigin::sWVR;
    
    case CalDataOriginMod::CHANNEL_AVERAGE_AUTO:
      return CCalDataOrigin::sCHANNEL_AVERAGE_AUTO;
    
    case CalDataOriginMod::CHANNEL_AVERAGE_CROSS:
      return CCalDataOrigin::sCHANNEL_AVERAGE_CROSS;
    
    case CalDataOriginMod::FULL_RESOLUTION_AUTO:
      return CCalDataOrigin::sFULL_RESOLUTION_AUTO;
    
    case CalDataOriginMod::FULL_RESOLUTION_CROSS:
      return CCalDataOrigin::sFULL_RESOLUTION_CROSS;
    
    case CalDataOriginMod::OPTICAL_POINTING:
      return CCalDataOrigin::sOPTICAL_POINTING;
    
    case CalDataOriginMod::HOLOGRAPHY:
      return CCalDataOrigin::sHOLOGRAPHY;
    	
    }
    return std::string("");
}

	

	
std::string CCalDataOrigin::help(const CalDataOriginMod::CalDataOrigin& f) {
    switch (f) {
    
    case CalDataOriginMod::TOTAL_POWER:
      return CCalDataOrigin::hTOTAL_POWER;
    
    case CalDataOriginMod::WVR:
      return CCalDataOrigin::hWVR;
    
    case CalDataOriginMod::CHANNEL_AVERAGE_AUTO:
      return CCalDataOrigin::hCHANNEL_AVERAGE_AUTO;
    
    case CalDataOriginMod::CHANNEL_AVERAGE_CROSS:
      return CCalDataOrigin::hCHANNEL_AVERAGE_CROSS;
    
    case CalDataOriginMod::FULL_RESOLUTION_AUTO:
      return CCalDataOrigin::hFULL_RESOLUTION_AUTO;
    
    case CalDataOriginMod::FULL_RESOLUTION_CROSS:
      return CCalDataOrigin::hFULL_RESOLUTION_CROSS;
    
    case CalDataOriginMod::OPTICAL_POINTING:
      return CCalDataOrigin::hOPTICAL_POINTING;
    
    case CalDataOriginMod::HOLOGRAPHY:
      return CCalDataOrigin::hHOLOGRAPHY;
    	
    }
    return std::string("");
}
   	

CalDataOriginMod::CalDataOrigin CCalDataOrigin::newCalDataOrigin(const std::string& name) {
		
    if (name == CCalDataOrigin::sTOTAL_POWER) {
        return CalDataOriginMod::TOTAL_POWER;
    }
    	
    if (name == CCalDataOrigin::sWVR) {
        return CalDataOriginMod::WVR;
    }
    	
    if (name == CCalDataOrigin::sCHANNEL_AVERAGE_AUTO) {
        return CalDataOriginMod::CHANNEL_AVERAGE_AUTO;
    }
    	
    if (name == CCalDataOrigin::sCHANNEL_AVERAGE_CROSS) {
        return CalDataOriginMod::CHANNEL_AVERAGE_CROSS;
    }
    	
    if (name == CCalDataOrigin::sFULL_RESOLUTION_AUTO) {
        return CalDataOriginMod::FULL_RESOLUTION_AUTO;
    }
    	
    if (name == CCalDataOrigin::sFULL_RESOLUTION_CROSS) {
        return CalDataOriginMod::FULL_RESOLUTION_CROSS;
    }
    	
    if (name == CCalDataOrigin::sOPTICAL_POINTING) {
        return CalDataOriginMod::OPTICAL_POINTING;
    }
    	
    if (name == CCalDataOrigin::sHOLOGRAPHY) {
        return CalDataOriginMod::HOLOGRAPHY;
    }
    
    throw badString(name);
}

CalDataOriginMod::CalDataOrigin CCalDataOrigin::literal(const std::string& name) {
		
    if (name == CCalDataOrigin::sTOTAL_POWER) {
        return CalDataOriginMod::TOTAL_POWER;
    }
    	
    if (name == CCalDataOrigin::sWVR) {
        return CalDataOriginMod::WVR;
    }
    	
    if (name == CCalDataOrigin::sCHANNEL_AVERAGE_AUTO) {
        return CalDataOriginMod::CHANNEL_AVERAGE_AUTO;
    }
    	
    if (name == CCalDataOrigin::sCHANNEL_AVERAGE_CROSS) {
        return CalDataOriginMod::CHANNEL_AVERAGE_CROSS;
    }
    	
    if (name == CCalDataOrigin::sFULL_RESOLUTION_AUTO) {
        return CalDataOriginMod::FULL_RESOLUTION_AUTO;
    }
    	
    if (name == CCalDataOrigin::sFULL_RESOLUTION_CROSS) {
        return CalDataOriginMod::FULL_RESOLUTION_CROSS;
    }
    	
    if (name == CCalDataOrigin::sOPTICAL_POINTING) {
        return CalDataOriginMod::OPTICAL_POINTING;
    }
    	
    if (name == CCalDataOrigin::sHOLOGRAPHY) {
        return CalDataOriginMod::HOLOGRAPHY;
    }
    
    throw badString(name);
}

CalDataOriginMod::CalDataOrigin CCalDataOrigin::from_int(unsigned int i) {
	vector<string> names = sCalDataOriginSet();
	if (i >= names.size()) throw badInt(i);
	return newCalDataOrigin(names.at(i));
}

	

string CCalDataOrigin::badString(const string& name) {
	return "'"+name+"' does not correspond to any literal in the enumeration 'CalDataOrigin'.";
}

string CCalDataOrigin::badInt(unsigned int i) {
	ostringstream oss ;
	oss << "'" << i << "' is out of range for the enumeration 'CalDataOrigin'.";
	return oss.str();
}
