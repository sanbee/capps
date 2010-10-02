
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
 * File PointingTable.cpp
 */
#include <ConversionException.h>
#include <DuplicateKey.h>
#include <OutOfBoundsException.h>

using asdm::ConversionException;
using asdm::DuplicateKey;
using asdm::OutOfBoundsException;

#include <ASDM.h>
#include <PointingTable.h>
#include <PointingRow.h>
#include <Parser.h>

using asdm::ASDM;
using asdm::PointingTable;
using asdm::PointingRow;
using asdm::Parser;

#include <EndianStream.h>
using asdm::EndianOSStream;
using asdm::EndianISStream;

#include <iostream>
#include <sstream>
#include <set>

#include <Misc.h>

using namespace std;
using namespace asdm;


namespace asdm {

	string PointingTable::tableName = "Pointing";

	/**
	 * The list of field names that make up key key.
	 * (Initialization is in the constructor.)
	 */
	vector<string> PointingTable::key;

	/**
	 * Return the list of field names that make up key key
	 * as an array of strings.
	 */	
	vector<string> PointingTable::getKeyName() {
		return key;
	}


	PointingTable::PointingTable(ASDM &c) : container(c) {

	
		key.push_back("antennaId");
	
		key.push_back("timeInterval");
	


		// Define a default entity.
		entity.setEntityId(EntityId("uid://X0/X0/X0"));
		entity.setEntityIdEncrypted("na");
		entity.setEntityTypeName("PointingTable");
		entity.setEntityVersion("1");
		entity.setInstanceVersion("1");

		// Archive binary
		archiveAsBin = true;
		
		// File binary
		fileAsBin = true;	
	}
	
/**
 * A destructor for PointingTable.
 */	

	
		
	PointingTable::~PointingTable() {
		map<string, TIME_ROWS >::iterator mapIter;
		vector<PointingRow *>::iterator rowIter;
		
		for(mapIter=context.begin(); mapIter!=context.end(); mapIter++)
			for(rowIter=((*mapIter).second).begin(); rowIter!=((*mapIter).second).end(); rowIter++)
				delete (*rowIter);
	}	
		
	


	/**
	 * Container to which this table belongs.
	 */
	ASDM &PointingTable::getContainer() const {
		return container;
	}

	/**
	 * Return the number of rows in the table.
	 */
	
	
		
	unsigned int PointingTable::size() {
		int result = 0;
		
		map<string, TIME_ROWS >::iterator mapIter;
		for (mapIter=context.begin(); mapIter!=context.end(); mapIter++) 
			result += ((*mapIter).second).size();
			
		return result;
	}	
		
	
	
	
	/**
	 * Return the name of this table.
	 */
	string PointingTable::getName() const {
		return tableName;
	}

	/**
	 * Return this table's Entity.
	 */
	Entity PointingTable::getEntity() const {
		return entity;
	}

	/**
	 * Set this table's Entity.
	 */
	void PointingTable::setEntity(Entity e) {
		this->entity = e; 
	}
	
	//
	// ====> Row creation.
	//
	
	/**
	 * Create a new row.
	 */
	PointingRow *PointingTable::newRow() {
		return new PointingRow (*this);
	}

	PointingRow *PointingTable::newRowEmpty() {
		return new PointingRow (*this);
	}

	/**
	 * Create a new row initialized to the specified values.
	 * @return a pointer on the created and initialized row.
	
 	 * @param antennaId. 
	
 	 * @param timeInterval. 
	
 	 * @param pointingModelId. 
	
 	 * @param numPoly. 
	
 	 * @param timeOrigin. 
	
 	 * @param pointingDirection. 
	
 	 * @param target. 
	
 	 * @param offset. 
	
 	 * @param encoder. 
	
 	 * @param pointingTracking. 
	
     */
	PointingRow* PointingTable::newRow(Tag antennaId, ArrayTimeInterval timeInterval, int pointingModelId, int numPoly, ArrayTime timeOrigin, vector<vector<Angle > > pointingDirection, vector<vector<Angle > > target, vector<vector<Angle > > offset, vector<Angle > encoder, bool pointingTracking){
		PointingRow *row = new PointingRow(*this);
			
		row->setAntennaId(antennaId);
			
		row->setTimeInterval(timeInterval);
			
		row->setPointingModelId(pointingModelId);
			
		row->setNumPoly(numPoly);
			
		row->setTimeOrigin(timeOrigin);
			
		row->setPointingDirection(pointingDirection);
			
		row->setTarget(target);
			
		row->setOffset(offset);
			
		row->setEncoder(encoder);
			
		row->setPointingTracking(pointingTracking);
	
		return row;		
	}	
	
	PointingRow* PointingTable::newRowFull(Tag antennaId, ArrayTimeInterval timeInterval, int pointingModelId, int numPoly, ArrayTime timeOrigin, vector<vector<Angle > > pointingDirection, vector<vector<Angle > > target, vector<vector<Angle > > offset, vector<Angle > encoder, bool pointingTracking){
		PointingRow *row = new PointingRow(*this);
			
		row->setAntennaId(antennaId);
			
		row->setTimeInterval(timeInterval);
			
		row->setPointingModelId(pointingModelId);
			
		row->setNumPoly(numPoly);
			
		row->setTimeOrigin(timeOrigin);
			
		row->setPointingDirection(pointingDirection);
			
		row->setTarget(target);
			
		row->setOffset(offset);
			
		row->setEncoder(encoder);
			
		row->setPointingTracking(pointingTracking);
	
		return row;		
	}		



PointingRow* PointingTable::newRow(PointingRow* row) {
	return new PointingRow(*this, *row);
}

PointingRow* PointingTable::newRowCopy(PointingRow* row) {
	return new PointingRow(*this, *row);
}
	//
	// Append a row to its table.
	//

	
	
		
		
	/** 
	 * Returns a string built by concatenating the ascii representation of the
	 * parameters values suffixed with a "_" character.
	 */
	 string PointingTable::Key(Tag antennaId) {
	 	ostringstream ostrstr;
	 		ostrstr  
			
				<<  antennaId.toString()  << "_"
			
			;
		return ostrstr.str();	 	
	 }

	PointingRow* PointingTable::add(PointingRow* x) {		 
		ArrayTime startTime = x->getTimeInterval().getStart();

		/*
	 	 * Is there already a context for this combination of not temporal 
	 	 * attributes ?
	 	 */
		string k = Key(
						x->getAntennaId()
					   );
 
		if (context.find(k) == context.end()) { 
			// There is not yet a context ...
			// Create and initialize an entry in the context map for this combination....
			TIME_ROWS v;
			context[k] = v;			
		}
		
		return insertByStartTime(x, context[k]);
	}	
		
	




	// 
	// A private method to append a row to its table, used by input conversion
	// methods.
	//

	
	
		
		
			
	PointingRow*  PointingTable::checkAndAdd(PointingRow* x) throw (DuplicateKey) {
		
		string k = Key( 
						x->getAntennaId() 
					   );
					   
		TIME_ROWS dummyRow;

		// Do we already have rows with that value of AntennaId
		if (context.find(k) == context.end()) {
			context[k] = dummyRow;
		}

		return insertByStartTime(x, context[k]);
	}
					
		







	

	
	
		
	/**
	 * Get all rows.
	 * @return Alls rows as an array of PointingRow
	 */
	 vector<PointingRow *> PointingTable::get() {
	    return privateRows;
	    
	 /*
	 	vector<PointingRow *> v;
	 	map<string, TIME_ROWS>::iterator mapIter;
	 	vector<PointingRow *>::iterator rowIter;
	 	
	 	for (mapIter=context.begin(); mapIter!=context.end(); mapIter++) {
	 		for (rowIter=((*mapIter).second).begin(); rowIter!=((*mapIter).second).end(); rowIter++) 
	 			v.push_back(*rowIter); 
	 	}
	 	
	 	return v;
	 */
	 }
	 
	 vector<PointingRow *> *PointingTable::getByContext(Tag antennaId) {
	  	string k = Key(antennaId);
 
	    if (context.find(k) == context.end()) return 0;
 	   else return &(context[k]);		
	}		
		
	


	
		
		
			
			
/*
 ** Returns a PointingRow* given a key.
 ** @return a pointer to the row having the key whose values are passed as parameters, or 0 if
 ** no row exists for that key.
 **
 */
 	PointingRow* PointingTable::getRowByKey(Tag antennaId, ArrayTimeInterval timeInterval)  {
 		string keystr = Key(antennaId);
 		vector<PointingRow *> row;
 		
 		if ( context.find(keystr)  == context.end()) return 0;
 		
 		row = context[keystr];
 		
 		// Is the vector empty...impossible in principle !
 		if (row.size() == 0) return 0;
 		
 		// Only one element in the vector
 		if (row.size() == 1) {
 			PointingRow* r = row.at(0);
 			if ( r->getTimeInterval().contains(timeInterval.getStart()))
 				return r;
 			else
 				return 0;
 		}
 		
 		// Optimizations
 		PointingRow* last = row.at(row.size()-1);
 		if (timeInterval.getStart().get() >= (last->getTimeInterval().getStart().get()+last->getTimeInterval().getDuration().get())) return 0;
 		
 		PointingRow* first = row.at(0);
 		if (timeInterval.getStart().get() < first->getTimeInterval().getStart().get()) return 0;
 		
 		
 		// More than one row 
 		// Let's use a dichotomy method for the general case..	
 		int k0 = 0;
 		int k1 = row.size() - 1;
 		PointingRow* r = 0;
 		while (k0!=k1) {

		 
		  // is the start time contained in the time interval of row #k0 ?
		  r = row.at(k0);
		  if (r->getTimeInterval().contains(timeInterval.getStart())) return r;
		 
		  // is the start time contained in the time interval of row #k1 ?		  
		  r = row.at(k1);
		  if (r->getTimeInterval().contains(timeInterval.getStart())) return r;
		  
		  // Are the rows #k0 and #k1 consecutive
		  // Then we know for sure that there is no row containing the start of timeInterval.
		  if (k1 == k0 + 1) return 0;
		  
		  r = row.at((k0+k1)/2);
		  if ( timeInterval.getStart().get() <= r->getTimeInterval().getStart().get())
		    k1 = (k0 + k1) / 2;
		  else
		    k0 = (k0 + k1) / 2;
		}
		return 0;	
	}			
			
		
		
		
	




#ifndef WITHOUT_ACS
	// Conversion Methods

	PointingTableIDL *PointingTable::toIDL() {
		PointingTableIDL *x = new PointingTableIDL ();
		unsigned int nrow = size();
		x->row.length(nrow);
		vector<PointingRow*> v = get();
		for (unsigned int i = 0; i < nrow; ++i) {
			x->row[i] = *(v[i]->toIDL());
		}
		return x;
	}
#endif
	
#ifndef WITHOUT_ACS
	void PointingTable::fromIDL(PointingTableIDL x) throw(DuplicateKey,ConversionException) {
		unsigned int nrow = x.row.length();
		for (unsigned int i = 0; i < nrow; ++i) {
			PointingRow *tmp = newRow();
			tmp->setFromIDL(x.row[i]);
			// checkAndAdd(tmp);
			add(tmp);
		}
	}
#endif

	char *PointingTable::toFITS() const throw(ConversionException) {
		throw ConversionException("Not implemented","Pointing");
	}

	void PointingTable::fromFITS(char *fits) throw(ConversionException) {
		throw ConversionException("Not implemented","Pointing");
	}

	string PointingTable::toVOTable() const throw(ConversionException) {
		throw ConversionException("Not implemented","Pointing");
	}

	void PointingTable::fromVOTable(string vo) throw(ConversionException) {
		throw ConversionException("Not implemented","Pointing");
	}

	string PointingTable::toXML()  throw(ConversionException) {
		string buf;
		buf.append("<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?> ");
//		buf.append("<PointingTable xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:noNamespaceSchemaLocation=\"../../idl/PointingTable.xsd\"> ");
		buf.append("<?xml-stylesheet type=\"text/xsl\" href=\"../asdm2html/table2html.xsl\"?> ");		
		buf.append("<PointingTable> ");
		buf.append(entity.toXML());
		string s = container.getEntity().toXML();
		// Change the "Entity" tag to "ContainerEntity".
		buf.append("<Container" + s.substr(1,s.length() - 1)+" ");
		vector<PointingRow*> v = get();
		for (unsigned int i = 0; i < v.size(); ++i) {
			try {
				buf.append(v[i]->toXML());
			} catch (NoSuchRow e) {
			}
			buf.append("  ");
		}		
		buf.append("</PointingTable> ");
		return buf;
	}
	
	void PointingTable::fromXML(string xmlDoc) throw(ConversionException) {
		Parser xml(xmlDoc);
		if (!xml.isStr("<PointingTable")) 
			error();
		// cout << "Parsing a PointingTable" << endl;
		string s = xml.getElement("<Entity","/>");
		if (s.length() == 0) 
			error();
		Entity e;
		e.setFromXML(s);
		if (e.getEntityTypeName() != "PointingTable")
			error();
		setEntity(e);
		// Skip the container's entity; but, it has to be there.
		s = xml.getElement("<ContainerEntity","/>");
		if (s.length() == 0) 
			error();

		// Get each row in the table.
		s = xml.getElementContent("<row>","</row>");
		PointingRow *row;
		while (s.length() != 0) {
			// cout << "Parsing a PointingRow" << endl; 
			row = newRow();
			row->setFromXML(s);
			try {
				checkAndAdd(row);
			} catch (DuplicateKey e1) {
				throw ConversionException(e1.getMessage(),"PointingTable");
			} 
			catch (UniquenessViolationException e1) {
				throw ConversionException(e1.getMessage(),"PointingTable");	
			}
			catch (...) {
				// cout << "Unexpected error in PointingTable::checkAndAdd called from PointingTable::fromXML " << endl;
			}
			s = xml.getElementContent("<row>","</row>");
		}
		if (!xml.isStr("</PointingTable>")) 
			error();
	}
	
	string PointingTable::toMIME() {		
		EndianOSStream eoss;
		
		string UID = getEntity().getEntityId().toString();
		string execBlockUID = getContainer().getEntity().getEntityId().toString();
		
		// The MIME Header
		eoss <<"MIME-Version: 1.0";
		eoss << "\n";
		eoss << "Content-Type: Multipart/Related; boundary='MIME_boundary'; type='text/xml'; start= '<header.xml>'";
		eoss <<"\n";
		eoss <<"Content-Description: Correlator";
		eoss <<"\n";
		eoss <<"alma-uid:" << UID;
		eoss <<"\n";
		eoss <<"\n";		
		
		// The MIME XML part header.
		eoss <<"--MIME_boundary";
		eoss <<"\n";
		eoss <<"Content-Type: text/xml; charset='ISO-8859-1'";
		eoss <<"\n";
		eoss <<"Content-Transfer-Encoding: 8bit";
		eoss <<"\n";
		eoss <<"Content-ID: <header.xml>";
		eoss <<"\n";
		eoss <<"\n";
		
		// The MIME XML part content.
		eoss << "<?xml version='1.0'  encoding='ISO-8859-1'?>";
		eoss << "\n";
		eoss<< "<ASDMBinaryTable  xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'  xsi:noNamespaceSchemaLocation='ASDMBinaryTable.xsd' ID='None'  version='1.0'>\n";
		eoss << "<ExecBlockUID>\n";
		eoss << execBlockUID  << "\n";
		eoss << "</ExecBlockUID>\n";
		eoss << "</ASDMBinaryTable>\n";		

		// The MIME binary part header
		eoss <<"--MIME_boundary";
		eoss <<"\n";
		eoss <<"Content-Type: binary/octet-stream";
		eoss <<"\n";
		eoss <<"Content-ID: <content.bin>";
		eoss <<"\n";
		eoss <<"\n";	
		
		// The MIME binary content
		entity.toBin(eoss);
		container.getEntity().toBin(eoss);
		eoss.writeInt((int) privateRows.size());
		for (unsigned int i = 0; i < privateRows.size(); i++) {
			privateRows.at(i)->toBin(eoss);	
		}
		
		// The closing MIME boundary
		eoss << "\n--MIME_boundary--";
		eoss << "\n";
		
		return eoss.str();	
	}

	void PointingTable::setFromMIME(const string & mimeMsg) {
		// cout << "Entering setFromMIME" << endl;
	 	string terminator = "Content-Type: binary/octet-stream\nContent-ID: <content.bin>\n\n";
	 	
	 	// Look for the string announcing the binary part.
	 	string::size_type loc = mimeMsg.find( terminator, 0 );
	 	
	 	if ( loc == string::npos ) {
	 		throw ConversionException("Failed to detect the beginning of the binary part", "Pointing");
	 	}
	
	 	// Create an EndianISStream from the substring containing the binary part.
	 	EndianISStream eiss(mimeMsg.substr(loc+terminator.size()));
	 	
	 	entity = Entity::fromBin(eiss);
	 	
	 	// We do nothing with that but we have to read it.
	 	Entity containerEntity = Entity::fromBin(eiss);
	 		 	
	 	int numRows = eiss.readInt();
	 	try {
	 		for (int i = 0; i < numRows; i++) {
	 			PointingRow* aRow = PointingRow::fromBin(eiss, *this);
	 			checkAndAdd(aRow);
	 		}
	 	}
	 	catch (DuplicateKey e) {
	 		throw ConversionException("Error while writing binary data , the message was "
	 					+ e.getMessage(), "Pointing");
	 	}
		catch (TagFormatException e) {
			throw ConversionException("Error while reading binary data , the message was "
					+ e.getMessage(), "Pointing");
		} 		 	
	 }
	 
	void PointingTable::error() throw(ConversionException) {
		throw ConversionException("Invalid xml document","Pointing");
	}
	
	void PointingTable::toFile(string directory) {
		if (!directoryExists(directory.c_str()) &&
			!createPath(directory.c_str())) {
			throw ConversionException("Could not create directory " , directory);
		}
		
		if (fileAsBin) {
			// write the bin serialized
			string fileName = directory + "/Pointing.bin";
			ofstream tableout(fileName.c_str(),ios::out|ios::trunc);
			if (tableout.rdstate() == ostream::failbit)
				throw ConversionException("Could not open file " + fileName + " to write ", "Pointing");
			tableout << toMIME() << endl;
			tableout.close();
			if (tableout.rdstate() == ostream::failbit)
				throw ConversionException("Could not close file " + fileName, "Pointing");
		}
		else {
			// write the XML
			string fileName = directory + "/Pointing.xml";
			ofstream tableout(fileName.c_str(),ios::out|ios::trunc);
			if (tableout.rdstate() == ostream::failbit)
				throw ConversionException("Could not open file " + fileName + " to write ", "Pointing");
			tableout << toXML() << endl;
			tableout.close();
			if (tableout.rdstate() == ostream::failbit)
				throw ConversionException("Could not close file " + fileName, "Pointing");
		}
	}		

	void PointingTable::setFromFile(const string& directory) {
		string tablename;
		if (fileAsBin)
			tablename = directory + "/Pointing.bin";
		else
			tablename = directory + "/Pointing.xml";
			
		// Determine the file size.
		ifstream::pos_type size;
		ifstream tablefile(tablename.c_str(), ios::in|ios::binary|ios::ate);

 		if (tablefile.is_open()) { 
  				size = tablefile.tellg(); 
  		}
		else {
				throw ConversionException("Could not open file " + tablename, "Pointing");
		}
		
		// Re position to the beginning.
		tablefile.seekg(0);
		
		// Read in a stringstream.
		stringstream ss;
		ss << tablefile.rdbuf();

		if (tablefile.rdstate() == istream::failbit || tablefile.rdstate() == istream::badbit) {
			throw ConversionException("Error reading file " + tablename,"Pointing");
		}

		// And close
		tablefile.close();
		if (tablefile.rdstate() == istream::failbit)
			throw ConversionException("Could not close file " + tablename,"Pointing");
					
		// And parse the content with the appropriate method
		if (fileAsBin) 
			setFromMIME(ss.str());
		else
			fromXML(ss.str());				
	}
						
	/**
	 * Insert a PointingRow* in a vector of PointingRow* so that it's ordered by ascending start time.
	 *
	 * @param PointingRow* x . The pointer to be inserted.
	 * @param vector <PointingRow*>& row. A reference to the vector where to insert x.
	 *
	 */
	 PointingRow* PointingTable::insertByStartTime(PointingRow* x, vector<PointingRow*>& row) {
				
		vector <PointingRow*>::iterator theIterator;
		
		ArrayTime start = x->timeInterval.getStart();

		// cout << "row size " << row.size() << endl;
    	// Is the row vector empty ?
    	if (row.size() == 0) {
    		row.push_back(x);
    		privateRows.push_back(x);
    		x->isAdded();
    		return x;
    	}
    	
    	// Optimization for the case of insertion by ascending time.
    	PointingRow* last = *(row.end()-1);
        
    	if ( start > last->timeInterval.getStart() ) {
	  //
	  // Modify the duration of last if and only if the start time of x
	  // is located strictly before the end time of last.
	  // 
	  if ( start < (last->timeInterval.getStart() + last->timeInterval.getDuration()))
	    last->timeInterval.setDuration(start - last->timeInterval.getStart());
	  row.push_back(x);
	  privateRows.push_back(x);
	  x->isAdded();
	  return x;
    	}
    	
    	// Optimization for the case of insertion by descending time.
    	PointingRow* first = *(row.begin());
        
    	if ( start < first->timeInterval.getStart() ) {
	  //
	  // Modify the duration of x if and only if the start time of first
	  // is located strictly before the end time of x.
	  //
	  if ( first->timeInterval.getStart() < (start + x->timeInterval.getDuration()) )
	    x->timeInterval.setDuration(first->timeInterval.getStart() - start);
	  row.insert(row.begin(), x);
	  privateRows.push_back(x);
	  x->isAdded();
	  return x;
    	}
    	
    	// Case where x has to be inserted inside row; let's use a dichotomy
    	// method to find the insertion index.
		unsigned int k0 = 0;
		unsigned int k1 = row.size() - 1;
	
		while (k0 != (k1 - 1)) {
			if (start == row[k0]->timeInterval.getStart()) {
				if (row[k0]->equalByRequiredValue(x))
					return row[k0];
				else
					throw DuplicateKey("DuplicateKey exception in ", "PointingTable");	
			}
			else if (start == row[k1]->timeInterval.getStart()) {
				if (row[k1]->equalByRequiredValue(x))
					return row[k1];
				else
					throw DuplicateKey("DuplicateKey exception in ", "PointingTable");	
			}
			else {
				if (start <= row[(k0+k1)/2]->timeInterval.getStart())
					k1 = (k0 + k1) / 2;
				else
					k0 = (k0 + k1) / 2;				
			} 	
		}
	
		row[k0]->timeInterval.setDuration(start-row[k0]->timeInterval.getStart());
		x->timeInterval.setDuration(row[k0+1]->timeInterval.getStart() - start);
		row.insert(row.begin()+(k0+1), x);
		privateRows.push_back(x);
   		x->isAdded();
		return x;   
    }   	
	
	

	
} // End namespace asdm
 
