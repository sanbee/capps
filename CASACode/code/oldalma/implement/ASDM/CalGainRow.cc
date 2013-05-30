
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
 * Warning!
 *  -------------------------------------------------------------------- 
 * | This is generated code!  Do not modify this file.                  |
 * | If you do, all changes will be lost when the file is re-generated. |
 *  --------------------------------------------------------------------
 *
 * File CalGainRow.cpp
 */
 
#include <vector>
using std::vector;

#include <set>
using std::set;

#include <ASDM.h>
#include <CalGainRow.h>
#include <CalGainTable.h>

#include <CalDataTable.h>
#include <CalDataRow.h>

#include <CalReductionTable.h>
#include <CalReductionRow.h>
	

using asdm::ASDM;
using asdm::CalGainRow;
using asdm::CalGainTable;

using asdm::CalDataTable;
using asdm::CalDataRow;

using asdm::CalReductionTable;
using asdm::CalReductionRow;


#include <Parser.h>
using asdm::Parser;

#include <EnumerationParser.h>
 
#include <InvalidArgumentException.h>
using asdm::InvalidArgumentException;

namespace asdm {

	CalGainRow::~CalGainRow() {
	}

	/**
	 * Return the table to which this row belongs.
	 */
	CalGainTable &CalGainRow::getTable() const {
		return table;
	}
	
	void CalGainRow::isAdded() {
		hasBeenAdded = true;
	}
	
	
#ifndef WITHOUT_ACS
	/**
	 * Return this row in the form of an IDL struct.
	 * @return The values of this row as a CalGainRowIDL struct.
	 */
	CalGainRowIDL *CalGainRow::toIDL() const {
		CalGainRowIDL *x = new CalGainRowIDL ();
		
		// Fill the IDL structure.
	
		
	
  		
		
		
			
		x->startValidTime = startValidTime.toIDLArrayTime();
			
		
	

	
  		
		
		
			
		x->endValidTime = endValidTime.toIDLArrayTime();
			
		
	

	
  		
		
		
			
		x->gain.length(gain.size());
		for (unsigned int i = 0; i < gain.size(); i++) {
			x->gain[i].length(gain.at(i).size());			 		
		}
		
		for (unsigned int i = 0; i < gain.size() ; i++)
			for (unsigned int j = 0; j < gain.at(i).size(); j++)
					
						
				x->gain[i][j] = gain.at(i).at(j);
		 				
			 						
		
			
		
	

	
  		
		
		
			
		x->gainValid.length(gainValid.size());
		for (unsigned int i = 0; i < gainValid.size(); ++i) {
			
				
			x->gainValid[i] = gainValid.at(i);
	 			
	 		
	 	}
			
		
	

	
  		
		
		
			
		x->fit.length(fit.size());
		for (unsigned int i = 0; i < fit.size(); i++) {
			x->fit[i].length(fit.at(i).size());			 		
		}
		
		for (unsigned int i = 0; i < fit.size() ; i++)
			for (unsigned int j = 0; j < fit.at(i).size(); j++)
					
						
				x->fit[i][j] = fit.at(i).at(j);
		 				
			 						
		
			
		
	

	
  		
		
		
			
		x->fitWeight.length(fitWeight.size());
		for (unsigned int i = 0; i < fitWeight.size(); ++i) {
			
				
			x->fitWeight[i] = fitWeight.at(i);
	 			
	 		
	 	}
			
		
	

	
  		
		
		
			
				
		x->totalGainValid = totalGainValid;
 				
 			
		
	

	
  		
		
		
			
				
		x->totalFit = totalFit;
 				
 			
		
	

	
  		
		
		
			
				
		x->totalFitWeight = totalFitWeight;
 				
 			
		
	

	
	
		
	
  	
 		
		
	 	
			
		x->calDataId = calDataId.toIDLTag();
			
	 	 		
  	

	
  	
 		
		
	 	
			
		x->calReductionId = calReductionId.toIDLTag();
			
	 	 		
  	

	
		
	

	

		
		return x;
	
	}
#endif
	

#ifndef WITHOUT_ACS
	/**
	 * Fill the values of this row from the IDL struct CalGainRowIDL.
	 * @param x The IDL struct containing the values used to fill this row.
	 */
	void CalGainRow::setFromIDL (CalGainRowIDL x) throw(ConversionException) {
		try {
		// Fill the values from x.
	
		
	
		
		
			
		setStartValidTime(ArrayTime (x.startValidTime));
			
 		
		
	

	
		
		
			
		setEndValidTime(ArrayTime (x.endValidTime));
			
 		
		
	

	
		
		
			
		gain .clear();
		vector<float> v_aux_gain;
		for (unsigned int i = 0; i < x.gain.length(); ++i) {
			v_aux_gain.clear();
			for (unsigned int j = 0; j < x.gain[0].length(); ++j) {
				
				v_aux_gain.push_back(x.gain[i][j]);
	  			
  			}
  			gain.push_back(v_aux_gain);			
		}
			
  		
		
	

	
		
		
			
		gainValid .clear();
		for (unsigned int i = 0; i <x.gainValid.length(); ++i) {
			
			gainValid.push_back(x.gainValid[i]);
  			
		}
			
  		
		
	

	
		
		
			
		fit .clear();
		vector<float> v_aux_fit;
		for (unsigned int i = 0; i < x.fit.length(); ++i) {
			v_aux_fit.clear();
			for (unsigned int j = 0; j < x.fit[0].length(); ++j) {
				
				v_aux_fit.push_back(x.fit[i][j]);
	  			
  			}
  			fit.push_back(v_aux_fit);			
		}
			
  		
		
	

	
		
		
			
		fitWeight .clear();
		for (unsigned int i = 0; i <x.fitWeight.length(); ++i) {
			
			fitWeight.push_back(x.fitWeight[i]);
  			
		}
			
  		
		
	

	
		
		
			
		setTotalGainValid(x.totalGainValid);
  			
 		
		
	

	
		
		
			
		setTotalFit(x.totalFit);
  			
 		
		
	

	
		
		
			
		setTotalFitWeight(x.totalFitWeight);
  			
 		
		
	

	
	
		
	
		
		
			
		setCalDataId(Tag (x.calDataId));
			
 		
		
	

	
		
		
			
		setCalReductionId(Tag (x.calReductionId));
			
 		
		
	

	
		
	

	

		} catch (IllegalAccessException err) {
			throw new ConversionException (err.getMessage(),"CalGain");
		}
	}
#endif
	
	/**
	 * Return this row in the form of an XML string.
	 * @return The values of this row as an XML string.
	 */
	string CalGainRow::toXML() const {
		string buf;
		buf.append("<row> \n");
		
	
		
  	
 		
		
		Parser::toXML(startValidTime, "startValidTime", buf);
		
		
	

  	
 		
		
		Parser::toXML(endValidTime, "endValidTime", buf);
		
		
	

  	
 		
		
		Parser::toXML(gain, "gain", buf);
		
		
	

  	
 		
		
		Parser::toXML(gainValid, "gainValid", buf);
		
		
	

  	
 		
		
		Parser::toXML(fit, "fit", buf);
		
		
	

  	
 		
		
		Parser::toXML(fitWeight, "fitWeight", buf);
		
		
	

  	
 		
		
		Parser::toXML(totalGainValid, "totalGainValid", buf);
		
		
	

  	
 		
		
		Parser::toXML(totalFit, "totalFit", buf);
		
		
	

  	
 		
		
		Parser::toXML(totalFitWeight, "totalFitWeight", buf);
		
		
	

	
	
		
  	
 		
		
		Parser::toXML(calDataId, "calDataId", buf);
		
		
	

  	
 		
		
		Parser::toXML(calReductionId, "calReductionId", buf);
		
		
	

	
		
	

	

		
		buf.append("</row>\n");
		return buf;
	}

	/**
	 * Fill the values of this row from an XML string 
	 * that was produced by the toXML() method.
	 * @param x The XML string being used to set the values of this row.
	 */
	void CalGainRow::setFromXML (string rowDoc) throw(ConversionException) {
		Parser row(rowDoc);
		string s = "";
		try {
	
		
	
  		
			
	  	setStartValidTime(Parser::getArrayTime("startValidTime","CalGain",rowDoc));
			
		
	

	
  		
			
	  	setEndValidTime(Parser::getArrayTime("endValidTime","CalGain",rowDoc));
			
		
	

	
  		
			
					
	  	setGain(Parser::get2DFloat("gain","CalGain",rowDoc));
	  			
	  		
		
	

	
  		
			
					
	  	setGainValid(Parser::get1DBoolean("gainValid","CalGain",rowDoc));
	  			
	  		
		
	

	
  		
			
					
	  	setFit(Parser::get2DFloat("fit","CalGain",rowDoc));
	  			
	  		
		
	

	
  		
			
					
	  	setFitWeight(Parser::get1DFloat("fitWeight","CalGain",rowDoc));
	  			
	  		
		
	

	
  		
			
	  	setTotalGainValid(Parser::getBoolean("totalGainValid","CalGain",rowDoc));
			
		
	

	
  		
			
	  	setTotalFit(Parser::getFloat("totalFit","CalGain",rowDoc));
			
		
	

	
  		
			
	  	setTotalFitWeight(Parser::getFloat("totalFitWeight","CalGain",rowDoc));
			
		
	

	
	
		
	
  		
			
	  	setCalDataId(Parser::getTag("calDataId","CalData",rowDoc));
			
		
	

	
  		
			
	  	setCalReductionId(Parser::getTag("calReductionId","CalReduction",rowDoc));
			
		
	

	
		
	

	

		} catch (IllegalAccessException err) {
			throw ConversionException (err.getMessage(),"CalGain");
		}
	}
	
	////////////////////////////////
	// Intrinsic Table Attributes //
	////////////////////////////////
	
	

	
 	/**
 	 * Get startValidTime.
 	 * @return startValidTime as ArrayTime
 	 */
 	ArrayTime CalGainRow::getStartValidTime() const {
	
  		return startValidTime;
 	}

 	/**
 	 * Set startValidTime with the specified ArrayTime.
 	 * @param startValidTime The ArrayTime value to which startValidTime is to be set.
 	 
 	
 		
 	 */
 	void CalGainRow::setStartValidTime (ArrayTime startValidTime)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->startValidTime = startValidTime;
	
 	}
	
	

	

	
 	/**
 	 * Get endValidTime.
 	 * @return endValidTime as ArrayTime
 	 */
 	ArrayTime CalGainRow::getEndValidTime() const {
	
  		return endValidTime;
 	}

 	/**
 	 * Set endValidTime with the specified ArrayTime.
 	 * @param endValidTime The ArrayTime value to which endValidTime is to be set.
 	 
 	
 		
 	 */
 	void CalGainRow::setEndValidTime (ArrayTime endValidTime)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->endValidTime = endValidTime;
	
 	}
	
	

	

	
 	/**
 	 * Get gain.
 	 * @return gain as vector<vector<float > >
 	 */
 	vector<vector<float > > CalGainRow::getGain() const {
	
  		return gain;
 	}

 	/**
 	 * Set gain with the specified vector<vector<float > >.
 	 * @param gain The vector<vector<float > > value to which gain is to be set.
 	 
 	
 		
 	 */
 	void CalGainRow::setGain (vector<vector<float > > gain)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->gain = gain;
	
 	}
	
	

	

	
 	/**
 	 * Get gainValid.
 	 * @return gainValid as vector<bool >
 	 */
 	vector<bool > CalGainRow::getGainValid() const {
	
  		return gainValid;
 	}

 	/**
 	 * Set gainValid with the specified vector<bool >.
 	 * @param gainValid The vector<bool > value to which gainValid is to be set.
 	 
 	
 		
 	 */
 	void CalGainRow::setGainValid (vector<bool > gainValid)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->gainValid = gainValid;
	
 	}
	
	

	

	
 	/**
 	 * Get fit.
 	 * @return fit as vector<vector<float > >
 	 */
 	vector<vector<float > > CalGainRow::getFit() const {
	
  		return fit;
 	}

 	/**
 	 * Set fit with the specified vector<vector<float > >.
 	 * @param fit The vector<vector<float > > value to which fit is to be set.
 	 
 	
 		
 	 */
 	void CalGainRow::setFit (vector<vector<float > > fit)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->fit = fit;
	
 	}
	
	

	

	
 	/**
 	 * Get fitWeight.
 	 * @return fitWeight as vector<float >
 	 */
 	vector<float > CalGainRow::getFitWeight() const {
	
  		return fitWeight;
 	}

 	/**
 	 * Set fitWeight with the specified vector<float >.
 	 * @param fitWeight The vector<float > value to which fitWeight is to be set.
 	 
 	
 		
 	 */
 	void CalGainRow::setFitWeight (vector<float > fitWeight)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->fitWeight = fitWeight;
	
 	}
	
	

	

	
 	/**
 	 * Get totalGainValid.
 	 * @return totalGainValid as bool
 	 */
 	bool CalGainRow::getTotalGainValid() const {
	
  		return totalGainValid;
 	}

 	/**
 	 * Set totalGainValid with the specified bool.
 	 * @param totalGainValid The bool value to which totalGainValid is to be set.
 	 
 	
 		
 	 */
 	void CalGainRow::setTotalGainValid (bool totalGainValid)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->totalGainValid = totalGainValid;
	
 	}
	
	

	

	
 	/**
 	 * Get totalFit.
 	 * @return totalFit as float
 	 */
 	float CalGainRow::getTotalFit() const {
	
  		return totalFit;
 	}

 	/**
 	 * Set totalFit with the specified float.
 	 * @param totalFit The float value to which totalFit is to be set.
 	 
 	
 		
 	 */
 	void CalGainRow::setTotalFit (float totalFit)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->totalFit = totalFit;
	
 	}
	
	

	

	
 	/**
 	 * Get totalFitWeight.
 	 * @return totalFitWeight as float
 	 */
 	float CalGainRow::getTotalFitWeight() const {
	
  		return totalFitWeight;
 	}

 	/**
 	 * Set totalFitWeight with the specified float.
 	 * @param totalFitWeight The float value to which totalFitWeight is to be set.
 	 
 	
 		
 	 */
 	void CalGainRow::setTotalFitWeight (float totalFitWeight)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->totalFitWeight = totalFitWeight;
	
 	}
	
	

	
	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	

	
 	/**
 	 * Get calDataId.
 	 * @return calDataId as Tag
 	 */
 	Tag CalGainRow::getCalDataId() const {
	
  		return calDataId;
 	}

 	/**
 	 * Set calDataId with the specified Tag.
 	 * @param calDataId The Tag value to which calDataId is to be set.
 	 
 	
 		
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 	
 	 */
 	void CalGainRow::setCalDataId (Tag calDataId)  {
  	
  	
  		if (hasBeenAdded) {
 		
			throw IllegalAccessException("calDataId", "CalGain");
		
  		}
  	
 		this->calDataId = calDataId;
	
 	}
	
	

	

	
 	/**
 	 * Get calReductionId.
 	 * @return calReductionId as Tag
 	 */
 	Tag CalGainRow::getCalReductionId() const {
	
  		return calReductionId;
 	}

 	/**
 	 * Set calReductionId with the specified Tag.
 	 * @param calReductionId The Tag value to which calReductionId is to be set.
 	 
 	
 		
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 	
 	 */
 	void CalGainRow::setCalReductionId (Tag calReductionId)  {
  	
  	
  		if (hasBeenAdded) {
 		
			throw IllegalAccessException("calReductionId", "CalGain");
		
  		}
  	
 		this->calReductionId = calReductionId;
	
 	}
	
	

	///////////
	// Links //
	///////////
	
	
	
	
		

	/**
	 * Returns the pointer to the row in the CalData table having CalData.calDataId == calDataId
	 * @return a CalDataRow*
	 * 
	 
	 */
	 CalDataRow* CalGainRow::getCalDataUsingCalDataId() {
	 
	 	return table.getContainer().getCalData().getRowByKey(calDataId);
	 }
	 

	

	
	
	
		

	/**
	 * Returns the pointer to the row in the CalReduction table having CalReduction.calReductionId == calReductionId
	 * @return a CalReductionRow*
	 * 
	 
	 */
	 CalReductionRow* CalGainRow::getCalReductionUsingCalReductionId() {
	 
	 	return table.getContainer().getCalReduction().getRowByKey(calReductionId);
	 }
	 

	

	
	/**
	 * Create a CalGainRow.
	 * <p>
	 * This constructor is private because only the
	 * table can create rows.  All rows know the table
	 * to which they belong.
	 * @param table The table to which this row belongs.
	 */ 
	CalGainRow::CalGainRow (CalGainTable &t) : table(t) {
		hasBeenAdded = false;
		
	
	

	

	

	

	

	

	

	

	

	
	

	

	
	
	
	

	

	

	

	

	

	

	

	
	
	}
	
	CalGainRow::CalGainRow (CalGainTable &t, CalGainRow &row) : table(t) {
		hasBeenAdded = false;
		
		if (&row == 0) {
	
	
	

	

	

	

	

	

	

	

	

	
	

	
		
		}
		else {
	
		
			calDataId = row.calDataId;
		
			calReductionId = row.calReductionId;
		
		
		
		
			startValidTime = row.startValidTime;
		
			endValidTime = row.endValidTime;
		
			gain = row.gain;
		
			gainValid = row.gainValid;
		
			fit = row.fit;
		
			fitWeight = row.fitWeight;
		
			totalGainValid = row.totalGainValid;
		
			totalFit = row.totalFit;
		
			totalFitWeight = row.totalFitWeight;
		
		
		
		
		}	
	}

	
	bool CalGainRow::compareNoAutoInc(Tag calDataId, Tag calReductionId, ArrayTime startValidTime, ArrayTime endValidTime, vector<vector<float > > gain, vector<bool > gainValid, vector<vector<float > > fit, vector<float > fitWeight, bool totalGainValid, float totalFit, float totalFitWeight) {
		bool result;
		result = true;
		
	
		
		result = result && (this->calDataId == calDataId);
		
		if (!result) return false;
	

	
		
		result = result && (this->calReductionId == calReductionId);
		
		if (!result) return false;
	

	
		
		result = result && (this->startValidTime == startValidTime);
		
		if (!result) return false;
	

	
		
		result = result && (this->endValidTime == endValidTime);
		
		if (!result) return false;
	

	
		
		result = result && (this->gain == gain);
		
		if (!result) return false;
	

	
		
		result = result && (this->gainValid == gainValid);
		
		if (!result) return false;
	

	
		
		result = result && (this->fit == fit);
		
		if (!result) return false;
	

	
		
		result = result && (this->fitWeight == fitWeight);
		
		if (!result) return false;
	

	
		
		result = result && (this->totalGainValid == totalGainValid);
		
		if (!result) return false;
	

	
		
		result = result && (this->totalFit == totalFit);
		
		if (!result) return false;
	

	
		
		result = result && (this->totalFitWeight == totalFitWeight);
		
		if (!result) return false;
	

		return result;
	}	
	
	
	
	bool CalGainRow::compareRequiredValue(ArrayTime startValidTime, ArrayTime endValidTime, vector<vector<float > > gain, vector<bool > gainValid, vector<vector<float > > fit, vector<float > fitWeight, bool totalGainValid, float totalFit, float totalFitWeight) {
		bool result;
		result = true;
		
	
		if (!(this->startValidTime == startValidTime)) return false;
	

	
		if (!(this->endValidTime == endValidTime)) return false;
	

	
		if (!(this->gain == gain)) return false;
	

	
		if (!(this->gainValid == gainValid)) return false;
	

	
		if (!(this->fit == fit)) return false;
	

	
		if (!(this->fitWeight == fitWeight)) return false;
	

	
		if (!(this->totalGainValid == totalGainValid)) return false;
	

	
		if (!(this->totalFit == totalFit)) return false;
	

	
		if (!(this->totalFitWeight == totalFitWeight)) return false;
	

		return result;
	}
	
	
	/**
	 * Return true if all required attributes of the value part are equal to their homologues
	 * in x and false otherwise.
	 *
	 * @param x a pointer on the CalGainRow whose required attributes of the value part 
	 * will be compared with those of this.
	 * @return a boolean.
	 */
	bool CalGainRow::equalByRequiredValue(CalGainRow* x) {
		
			
		if (this->startValidTime != x->startValidTime) return false;
			
		if (this->endValidTime != x->endValidTime) return false;
			
		if (this->gain != x->gain) return false;
			
		if (this->gainValid != x->gainValid) return false;
			
		if (this->fit != x->fit) return false;
			
		if (this->fitWeight != x->fitWeight) return false;
			
		if (this->totalGainValid != x->totalGainValid) return false;
			
		if (this->totalFit != x->totalFit) return false;
			
		if (this->totalFitWeight != x->totalFitWeight) return false;
			
		
		return true;
	}	
	

} // End namespace asdm
 