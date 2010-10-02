
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
 * File CFieldCode.cpp
 */
#include <sstream>
#include <CFieldCode.h>
#include <string>
using namespace std;

	
const std::string& CFieldCode::sNONE = "NONE";
	
const std::vector<std::string> CFieldCode::sFieldCodeSet() {
    std::vector<std::string> enumSet;
    
    enumSet.insert(enumSet.end(), CFieldCode::sNONE);
        
    return enumSet;
}

	

	
	
const std::string& CFieldCode::hNONE = "";
	
const std::vector<std::string> CFieldCode::hFieldCodeSet() {
    std::vector<std::string> enumSet;
    
    enumSet.insert(enumSet.end(), CFieldCode::hNONE);
        
    return enumSet;
}
   	

std::string CFieldCode::name(const FieldCodeMod::FieldCode& f) {
    switch (f) {
    
    case FieldCodeMod::NONE:
      return CFieldCode::sNONE;
    	
    }
    return std::string("");
}

	

	
std::string CFieldCode::help(const FieldCodeMod::FieldCode& f) {
    switch (f) {
    
    case FieldCodeMod::NONE:
      return CFieldCode::hNONE;
    	
    }
    return std::string("");
}
   	

FieldCodeMod::FieldCode CFieldCode::newFieldCode(const std::string& name) {
		
    if (name == CFieldCode::sNONE) {
        return FieldCodeMod::NONE;
    }
    
    throw badString(name);
}

FieldCodeMod::FieldCode CFieldCode::literal(const std::string& name) {
		
    if (name == CFieldCode::sNONE) {
        return FieldCodeMod::NONE;
    }
    
    throw badString(name);
}

FieldCodeMod::FieldCode CFieldCode::from_int(unsigned int i) {
	vector<string> names = sFieldCodeSet();
	if (i >= names.size()) throw badInt(i);
	return newFieldCode(names.at(i));
}

	

string CFieldCode::badString(const string& name) {
	return "'"+name+"' does not correspond to any literal in the enumeration 'FieldCode'.";
}

string CFieldCode::badInt(unsigned int i) {
	ostringstream oss ;
	oss << "'" << i << "' is out of range for the enumeration 'FieldCode'.";
	return oss.str();
}

