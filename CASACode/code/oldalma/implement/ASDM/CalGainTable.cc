
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
 * File CalGainTable.cpp
 */
#include <ConversionException.h>
#include <DuplicateKey.h>
#include <OutOfBoundsException.h>

using asdm::ConversionException;
using asdm::DuplicateKey;
using asdm::OutOfBoundsException;

#include <ASDM.h>
#include <CalGainTable.h>
#include <CalGainRow.h>
#include <Parser.h>

using asdm::ASDM;
using asdm::CalGainTable;
using asdm::CalGainRow;
using asdm::Parser;

#include <iostream>
#include <sstream>
#include <set>
using namespace std;

#include <Misc.h>
using namespace asdm;


namespace asdm {

	string CalGainTable::tableName = "CalGain";
	

	/**
	 * The list of field names that make up key key.
	 * (Initialization is in the constructor.)
	 */
	vector<string> CalGainTable::key;

	/**
	 * Return the list of field names that make up key key
	 * as an array of strings.
	 */	
	vector<string> CalGainTable::getKeyName() {
		return key;
	}


	CalGainTable::CalGainTable(ASDM &c) : container(c) {

	
		key.push_back("calDataId");
	
		key.push_back("calReductionId");
	


		// Define a default entity.
		entity.setEntityId(EntityId("uid://X0/X0/X0"));
		entity.setEntityIdEncrypted("na");
		entity.setEntityTypeName("CalGainTable");
		entity.setEntityVersion("1");
		entity.setInstanceVersion("1");
		
		// Archive XML
		archiveAsBin = false;
		
		// File XML
		fileAsBin = false;
	}
	
/**
 * A destructor for CalGainTable.
 */
 
	CalGainTable::~CalGainTable() {
		for (unsigned int i = 0; i < privateRows.size(); i++) 
			delete(privateRows.at(i));
	}

	/**
	 * Container to which this table belongs.
	 */
	ASDM &CalGainTable::getContainer() const {
		return container;
	}

	/**
	 * Return the number of rows in the table.
	 */

	unsigned int CalGainTable::size() {
		return row.size();
	}	
	
	
	/**
	 * Return the name of this table.
	 */
	string CalGainTable::getName() const {
		return tableName;
	}

	/**
	 * Return this table's Entity.
	 */
	Entity CalGainTable::getEntity() const {
		return entity;
	}

	/**
	 * Set this table's Entity.
	 */
	void CalGainTable::setEntity(Entity e) {
		this->entity = e; 
	}
	
	//
	// ====> Row creation.
	//
	
	/**
	 * Create a new row.
	 */
	CalGainRow *CalGainTable::newRow() {
		return new CalGainRow (*this);
	}
	
	CalGainRow *CalGainTable::newRowEmpty() {
		return newRow ();
	}


	/**
	 * Create a new row initialized to the specified values.
	 * @return a pointer on the created and initialized row.
	
 	 * @param calDataId. 
	
 	 * @param calReductionId. 
	
 	 * @param startValidTime. 
	
 	 * @param endValidTime. 
	
 	 * @param gain. 
	
 	 * @param gainValid. 
	
 	 * @param fit. 
	
 	 * @param fitWeight. 
	
 	 * @param totalGainValid. 
	
 	 * @param totalFit. 
	
 	 * @param totalFitWeight. 
	
     */
	CalGainRow* CalGainTable::newRow(Tag calDataId, Tag calReductionId, ArrayTime startValidTime, ArrayTime endValidTime, vector<vector<float > > gain, vector<bool > gainValid, vector<vector<float > > fit, vector<float > fitWeight, bool totalGainValid, float totalFit, float totalFitWeight){
		CalGainRow *row = new CalGainRow(*this);
			
		row->setCalDataId(calDataId);
			
		row->setCalReductionId(calReductionId);
			
		row->setStartValidTime(startValidTime);
			
		row->setEndValidTime(endValidTime);
			
		row->setGain(gain);
			
		row->setGainValid(gainValid);
			
		row->setFit(fit);
			
		row->setFitWeight(fitWeight);
			
		row->setTotalGainValid(totalGainValid);
			
		row->setTotalFit(totalFit);
			
		row->setTotalFitWeight(totalFitWeight);
	
		return row;		
	}	

	CalGainRow* CalGainTable::newRowFull(Tag calDataId, Tag calReductionId, ArrayTime startValidTime, ArrayTime endValidTime, vector<vector<float > > gain, vector<bool > gainValid, vector<vector<float > > fit, vector<float > fitWeight, bool totalGainValid, float totalFit, float totalFitWeight)	{
		CalGainRow *row = new CalGainRow(*this);
			
		row->setCalDataId(calDataId);
			
		row->setCalReductionId(calReductionId);
			
		row->setStartValidTime(startValidTime);
			
		row->setEndValidTime(endValidTime);
			
		row->setGain(gain);
			
		row->setGainValid(gainValid);
			
		row->setFit(fit);
			
		row->setFitWeight(fitWeight);
			
		row->setTotalGainValid(totalGainValid);
			
		row->setTotalFit(totalFit);
			
		row->setTotalFitWeight(totalFitWeight);
	
		return row;				
	}
	


CalGainRow* CalGainTable::newRow(CalGainRow* row) {
	return new CalGainRow(*this, *row);
}

CalGainRow* CalGainTable::newRowCopy(CalGainRow* row) {
	return new CalGainRow(*this, *row);
}

	//
	// Append a row to its table.
	//

	
	 
	/**
	 * Add a row.
	 * @throws DuplicateKey Thrown if the new row has a key that is already in the table.
	 * @param x A pointer to the row to be added.
	 * @return x
	 */
	CalGainRow* CalGainTable::add(CalGainRow* x) {
		
		if (getRowByKey(
						x->getCalDataId()
						,
						x->getCalReductionId()
						))
			//throw DuplicateKey(x.getCalDataId() + "|" + x.getCalReductionId(),"CalGain");
			throw DuplicateKey("Duplicate key exception in ","CalGainTable");
		
		row.push_back(x);
		privateRows.push_back(x);
		x->isAdded();
		return x;
	}

		





	// 
	// A private method to append a row to its table, used by input conversion
	// methods.
	//

	
	/**
	 * If this table has an autoincrementable attribute then check if *x verifies the rule of uniqueness and throw exception if not.
	 * Check if *x verifies the key uniqueness rule and throw an exception if not.
	 * Append x to its table.
	 * @param x a pointer on the row to be appended.
	 * @returns a pointer on x.
	 */
	CalGainRow*  CalGainTable::checkAndAdd(CalGainRow* x) throw (DuplicateKey) {
		
		
		if (getRowByKey(
	
			x->getCalDataId()
	,
			x->getCalReductionId()
			
		)) throw DuplicateKey("Duplicate key exception in ", "CalGainTable");
		
		row.push_back(x);
		privateRows.push_back(x);
		x->isAdded();
		return x;	
	}	







	

	//
	// ====> Methods returning rows.
	//	
	/**
	 * Get all rows.
	 * @return Alls rows as an array of CalGainRow
	 */
	vector<CalGainRow *> CalGainTable::get() {
		return privateRows;
		// return row;
	}

	
/*
 ** Returns a CalGainRow* given a key.
 ** @return a pointer to the row having the key whose values are passed as parameters, or 0 if
 ** no row exists for that key.
 **
 */
 	CalGainRow* CalGainTable::getRowByKey(Tag calDataId, Tag calReductionId)  {
	CalGainRow* aRow = 0;
	for (unsigned int i = 0; i < row.size(); i++) {
		aRow = row.at(i);
		
			
				if (aRow->calDataId != calDataId) continue;
			
		
			
				if (aRow->calReductionId != calReductionId) continue;
			
		
		return aRow;
	}
	return 0;		
}
	

	
/**
 * Look up the table for a row whose all attributes 
 * are equal to the corresponding parameters of the method.
 * @return a pointer on this row if any, 0 otherwise.
 *
			
 * @param calDataId.
 	 		
 * @param calReductionId.
 	 		
 * @param startValidTime.
 	 		
 * @param endValidTime.
 	 		
 * @param gain.
 	 		
 * @param gainValid.
 	 		
 * @param fit.
 	 		
 * @param fitWeight.
 	 		
 * @param totalGainValid.
 	 		
 * @param totalFit.
 	 		
 * @param totalFitWeight.
 	 		 
 */
CalGainRow* CalGainTable::lookup(Tag calDataId, Tag calReductionId, ArrayTime startValidTime, ArrayTime endValidTime, vector<vector<float > > gain, vector<bool > gainValid, vector<vector<float > > fit, vector<float > fitWeight, bool totalGainValid, float totalFit, float totalFitWeight) {
		CalGainRow* aRow;
		for (unsigned int i = 0; i < size(); i++) {
			aRow = row.at(i); 
			if (aRow->compareNoAutoInc(calDataId, calReductionId, startValidTime, endValidTime, gain, gainValid, fit, fitWeight, totalGainValid, totalFit, totalFitWeight)) return aRow;
		}			
		return 0;	
} 
	
 	 	

	





#ifndef WITHOUT_ACS
	// Conversion Methods

	CalGainTableIDL *CalGainTable::toIDL() {
		CalGainTableIDL *x = new CalGainTableIDL ();
		unsigned int nrow = size();
		x->row.length(nrow);
		vector<CalGainRow*> v = get();
		for (unsigned int i = 0; i < nrow; ++i) {
			x->row[i] = *(v[i]->toIDL());
		}
		return x;
	}
#endif
	
#ifndef WITHOUT_ACS
	void CalGainTable::fromIDL(CalGainTableIDL x) throw(DuplicateKey,ConversionException) {
		unsigned int nrow = x.row.length();
		for (unsigned int i = 0; i < nrow; ++i) {
			CalGainRow *tmp = newRow();
			tmp->setFromIDL(x.row[i]);
			// checkAndAdd(tmp);
			add(tmp);
		}
	}
#endif

	char *CalGainTable::toFITS() const throw(ConversionException) {
		throw ConversionException("Not implemented","CalGain");
	}

	void CalGainTable::fromFITS(char *fits) throw(ConversionException) {
		throw ConversionException("Not implemented","CalGain");
	}

	string CalGainTable::toVOTable() const throw(ConversionException) {
		throw ConversionException("Not implemented","CalGain");
	}

	void CalGainTable::fromVOTable(string vo) throw(ConversionException) {
		throw ConversionException("Not implemented","CalGain");
	}

	string CalGainTable::toXML()  throw(ConversionException) {
		string buf;
		buf.append("<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?> ");
//		buf.append("<CalGainTable xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:noNamespaceSchemaLocation=\"../../idl/CalGainTable.xsd\"> ");
		buf.append("<?xml-stylesheet type=\"text/xsl\" href=\"../asdm2html/table2html.xsl\"?> ");		
		buf.append("<CalGainTable> ");
		buf.append(entity.toXML());
		string s = container.getEntity().toXML();
		// Change the "Entity" tag to "ContainerEntity".
		buf.append("<Container" + s.substr(1,s.length() - 1)+" ");
		vector<CalGainRow*> v = get();
		for (unsigned int i = 0; i < v.size(); ++i) {
			try {
				buf.append(v[i]->toXML());
			} catch (NoSuchRow e) {
			}
			buf.append("  ");
		}		
		buf.append("</CalGainTable> ");
		return buf;
	}
	
	void CalGainTable::fromXML(string xmlDoc) throw(ConversionException) {
		Parser xml(xmlDoc);
		if (!xml.isStr("<CalGainTable")) 
			error();
		// cout << "Parsing a CalGainTable" << endl;
		string s = xml.getElement("<Entity","/>");
		if (s.length() == 0) 
			error();
		Entity e;
		e.setFromXML(s);
		if (e.getEntityTypeName() != "CalGainTable")
			error();
		setEntity(e);
		// Skip the container's entity; but, it has to be there.
		s = xml.getElement("<ContainerEntity","/>");
		if (s.length() == 0) 
			error();

		// Get each row in the table.
		s = xml.getElementContent("<row>","</row>");
		CalGainRow *row;
		while (s.length() != 0) {
			// cout << "Parsing a CalGainRow" << endl; 
			row = newRow();
			row->setFromXML(s);
			try {
				checkAndAdd(row);
			} catch (DuplicateKey e1) {
				throw ConversionException(e1.getMessage(),"CalGainTable");
			} 
			catch (UniquenessViolationException e1) {
				throw ConversionException(e1.getMessage(),"CalGainTable");	
			}
			catch (...) {
				// cout << "Unexpected error in CalGainTable::checkAndAdd called from CalGainTable::fromXML " << endl;
			}
			s = xml.getElementContent("<row>","</row>");
		}
		if (!xml.isStr("</CalGainTable>")) 
			error();
	}

	void CalGainTable::error() throw(ConversionException) {
		throw ConversionException("Invalid xml document","CalGain");
	}
	
	string CalGainTable::toMIME() {
	 // To be implemented
		return "";
	}
	
	void CalGainTable::setFromMIME(const string & mimeMsg) {
		// To be implemented
		;
	}
	
	
	void CalGainTable::toFile(string directory) {
		if (!directoryExists(directory.c_str()) &&
			!createPath(directory.c_str())) {
			throw ConversionException("Could not create directory " , directory);
		}
		
		if (fileAsBin) {
			// write the bin serialized
			string fileName = directory + "/CalGain.bin";
			ofstream tableout(fileName.c_str(),ios::out|ios::trunc);
			if (tableout.rdstate() == ostream::failbit)
				throw ConversionException("Could not open file " + fileName + " to write ", "CalGain");
			tableout << toMIME() << endl;
			tableout.close();
			if (tableout.rdstate() == ostream::failbit)
				throw ConversionException("Could not close file " + fileName, "CalGain");
		}
		else {
			// write the XML
			string fileName = directory + "/CalGain.xml";
			ofstream tableout(fileName.c_str(),ios::out|ios::trunc);
			if (tableout.rdstate() == ostream::failbit)
				throw ConversionException("Could not open file " + fileName + " to write ", "CalGain");
			tableout << toXML() << endl;
			tableout.close();
			if (tableout.rdstate() == ostream::failbit)
				throw ConversionException("Could not close file " + fileName, "CalGain");
		}
	}
	
	void CalGainTable::setFromFile(const string& directory) {
		string tablename;
		if (fileAsBin)
			tablename = directory + "/CalGain.bin";
		else
			tablename = directory + "/CalGain.xml";
			
		// Determine the file size.
		ifstream::pos_type size;
		ifstream tablefile(tablename.c_str(), ios::in|ios::binary|ios::ate);

 		if (tablefile.is_open()) { 
  				size = tablefile.tellg(); 
  		}
		else {
				throw ConversionException("Could not open file " + tablename, "CalGain");
		}
		
		// Re position to the beginning.
		tablefile.seekg(0);
		
		// Read in a stringstream.
		stringstream ss;
		ss << tablefile.rdbuf();

		if (tablefile.rdstate() == istream::failbit || tablefile.rdstate() == istream::badbit) {
			throw ConversionException("Error reading file " + tablename,"CalGain");
		}

		// And close
		tablefile.close();
		if (tablefile.rdstate() == istream::failbit)
			throw ConversionException("Could not close file " + tablename,"CalGain");
					
		// And parse the content with the appropriate method
		if (fileAsBin) 
			setFromMIME(ss.str());
		else
			fromXML(ss.str());	
	}			
			
	
	

	
} // End namespace asdm
 
