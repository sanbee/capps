
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
 * File CFluxCalibrationMethod.cpp
 */
#include <sstream>
#include <CFluxCalibrationMethod.h>
#include <string>
using namespace std;

	
const std::string& CFluxCalibrationMethod::sABSOLUTE = "ABSOLUTE";
	
const std::string& CFluxCalibrationMethod::sRELATIVE = "RELATIVE";
	
const std::string& CFluxCalibrationMethod::sEFFICIENCY = "EFFICIENCY";
	
const std::vector<std::string> CFluxCalibrationMethod::sFluxCalibrationMethodSet() {
    std::vector<std::string> enumSet;
    
    enumSet.insert(enumSet.end(), CFluxCalibrationMethod::sABSOLUTE);
    
    enumSet.insert(enumSet.end(), CFluxCalibrationMethod::sRELATIVE);
    
    enumSet.insert(enumSet.end(), CFluxCalibrationMethod::sEFFICIENCY);
        
    return enumSet;
}

	

	
	
const std::string& CFluxCalibrationMethod::hABSOLUTE = "Absolute flux calibration (based on standard antenna)";
	
const std::string& CFluxCalibrationMethod::hRELATIVE = "Relative flux calibration (based on a primary calibrator)";
	
const std::string& CFluxCalibrationMethod::hEFFICIENCY = "Flux calibrator based on tabulated antenna efficiciency";
	
const std::vector<std::string> CFluxCalibrationMethod::hFluxCalibrationMethodSet() {
    std::vector<std::string> enumSet;
    
    enumSet.insert(enumSet.end(), CFluxCalibrationMethod::hABSOLUTE);
    
    enumSet.insert(enumSet.end(), CFluxCalibrationMethod::hRELATIVE);
    
    enumSet.insert(enumSet.end(), CFluxCalibrationMethod::hEFFICIENCY);
        
    return enumSet;
}
   	

std::string CFluxCalibrationMethod::name(const FluxCalibrationMethodMod::FluxCalibrationMethod& f) {
    switch (f) {
    
    case FluxCalibrationMethodMod::ABSOLUTE:
      return CFluxCalibrationMethod::sABSOLUTE;
    
    case FluxCalibrationMethodMod::RELATIVE:
      return CFluxCalibrationMethod::sRELATIVE;
    
    case FluxCalibrationMethodMod::EFFICIENCY:
      return CFluxCalibrationMethod::sEFFICIENCY;
    	
    }
    return std::string("");
}

	

	
std::string CFluxCalibrationMethod::help(const FluxCalibrationMethodMod::FluxCalibrationMethod& f) {
    switch (f) {
    
    case FluxCalibrationMethodMod::ABSOLUTE:
      return CFluxCalibrationMethod::hABSOLUTE;
    
    case FluxCalibrationMethodMod::RELATIVE:
      return CFluxCalibrationMethod::hRELATIVE;
    
    case FluxCalibrationMethodMod::EFFICIENCY:
      return CFluxCalibrationMethod::hEFFICIENCY;
    	
    }
    return std::string("");
}
   	

FluxCalibrationMethodMod::FluxCalibrationMethod CFluxCalibrationMethod::newFluxCalibrationMethod(const std::string& name) {
		
    if (name == CFluxCalibrationMethod::sABSOLUTE) {
        return FluxCalibrationMethodMod::ABSOLUTE;
    }
    	
    if (name == CFluxCalibrationMethod::sRELATIVE) {
        return FluxCalibrationMethodMod::RELATIVE;
    }
    	
    if (name == CFluxCalibrationMethod::sEFFICIENCY) {
        return FluxCalibrationMethodMod::EFFICIENCY;
    }
    
    throw badString(name);
}

FluxCalibrationMethodMod::FluxCalibrationMethod CFluxCalibrationMethod::literal(const std::string& name) {
		
    if (name == CFluxCalibrationMethod::sABSOLUTE) {
        return FluxCalibrationMethodMod::ABSOLUTE;
    }
    	
    if (name == CFluxCalibrationMethod::sRELATIVE) {
        return FluxCalibrationMethodMod::RELATIVE;
    }
    	
    if (name == CFluxCalibrationMethod::sEFFICIENCY) {
        return FluxCalibrationMethodMod::EFFICIENCY;
    }
    
    throw badString(name);
}

FluxCalibrationMethodMod::FluxCalibrationMethod CFluxCalibrationMethod::from_int(unsigned int i) {
	vector<string> names = sFluxCalibrationMethodSet();
	if (i >= names.size()) throw badInt(i);
	return newFluxCalibrationMethod(names.at(i));
}

	

string CFluxCalibrationMethod::badString(const string& name) {
	return "'"+name+"' does not correspond to any literal in the enumeration 'FluxCalibrationMethod'.";
}

string CFluxCalibrationMethod::badInt(unsigned int i) {
	ostringstream oss ;
	oss << "'" << i << "' is out of range for the enumeration 'FluxCalibrationMethod'.";
	return oss.str();
}
