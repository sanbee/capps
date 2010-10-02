
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
 * File AnnotationRow.h
 */
 
#ifndef AnnotationRow_CLASS
#define AnnotationRow_CLASS

#include <vector>
#include <string>
#include <set>
using std::vector;
using std::string;
using std::set;

#ifndef WITHOUT_ACS
#include <asdmIDLC.h>
using asdmIDL::AnnotationRowIDL;
#endif

#include <Angle.h>
#include <AngularRate.h>
#include <ArrayTime.h>
#include <ArrayTimeInterval.h>
#include <Complex.h>
#include <Entity.h>
#include <EntityId.h>
#include <EntityRef.h>
#include <Flux.h>
#include <Frequency.h>
#include <Humidity.h>
#include <Interval.h>
#include <Length.h>
#include <Pressure.h>
#include <Speed.h>
#include <Tag.h>
#include <Temperature.h>
#include <ConversionException.h>
#include <NoSuchRow.h>
#include <IllegalAccessException.h>

/*
#include <Enumerations.h>
using namespace enumerations;
 */




	

	

	

	

	

	

	
#include "CBasebandName.h"
using namespace BasebandNameMod;
	

	

	

	

	

	

	

	



using asdm::Angle;
using asdm::AngularRate;
using asdm::ArrayTime;
using asdm::Complex;
using asdm::Entity;
using asdm::EntityId;
using asdm::EntityRef;
using asdm::Flux;
using asdm::Frequency;
using asdm::Humidity;
using asdm::Interval;
using asdm::Length;
using asdm::Pressure;
using asdm::Speed;
using asdm::Tag;
using asdm::Temperature;
using asdm::ConversionException;
using asdm::NoSuchRow;
using asdm::IllegalAccessException;

/*\file Annotation.h
    \brief Generated from model's revision "1.46", branch "HEAD"
*/

namespace asdm {

//class asdm::AnnotationTable;


// class asdm::AntennaRow;
class AntennaRow;
	

/**
 * The AnnotationRow class is a row of a AnnotationTable.
 * 
 * Generated from model's revision "1.46", branch "HEAD"
 *
 */
class AnnotationRow {
friend class asdm::AnnotationTable;

public:

	virtual ~AnnotationRow();

	/**
	 * Return the table to which this row belongs.
	 */
	AnnotationTable &getTable() const;
	
#ifndef WITHOUT_ACS
	/**
	 * Return this row in the form of an IDL struct.
	 * @return The values of this row as a AnnotationRowIDL struct.
	 */
	AnnotationRowIDL *toIDL() const;
#endif
	
#ifndef WITHOUT_ACS
	/**
	 * Fill the values of this row from the IDL struct AnnotationRowIDL.
	 * @param x The IDL struct containing the values used to fill this row.
	 */
	void setFromIDL (AnnotationRowIDL x) throw(ConversionException);
#endif
	
	/**
	 * Return this row in the form of an XML string.
	 * @return The values of this row as an XML string.
	 */
	string toXML() const;

	/**
	 * Fill the values of this row from an XML string 
	 * that was produced by the toXML() method.
	 * @param x The XML string being used to set the values of this row.
	 */
	void setFromXML (string rowDoc) throw(ConversionException);
	
	////////////////////////////////
	// Intrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute time
	
	
	

	
 	/**
 	 * Get time.
 	 * @return time as ArrayTime
 	 */
 	ArrayTime getTime() const;
	
 
 	
 	
 	/**
 	 * Set time with the specified ArrayTime.
 	 * @param time The ArrayTime value to which time is to be set.
 	 
 		
 			
 	 */
 	void setTime (ArrayTime time);
  		
	
	
	


	
	// ===> Attribute issue
	
	
	

	
 	/**
 	 * Get issue.
 	 * @return issue as string
 	 */
 	string getIssue() const;
	
 
 	
 	
 	/**
 	 * Set issue with the specified string.
 	 * @param issue The string value to which issue is to be set.
 	 
 		
 			
 	 */
 	void setIssue (string issue);
  		
	
	
	


	
	// ===> Attribute details
	
	
	

	
 	/**
 	 * Get details.
 	 * @return details as string
 	 */
 	string getDetails() const;
	
 
 	
 	
 	/**
 	 * Set details with the specified string.
 	 * @param details The string value to which details is to be set.
 	 
 		
 			
 	 */
 	void setDetails (string details);
  		
	
	
	


	
	// ===> Attribute interval, which is optional
	
	
	
	/**
	 * The attribute interval is optional. Return true if this attribute exists.
	 * @return true if and only if the interval attribute exists. 
	 */
	bool isIntervalExists() const;
	

	
 	/**
 	 * Get interval, which is optional.
 	 * @return interval as Interval
 	 * @throws IllegalAccessException If interval does not exist.
 	 */
 	Interval getInterval() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set interval with the specified Interval.
 	 * @param interval The Interval value to which interval is to be set.
 	 
 		
 	 */
 	void setInterval (Interval interval);
		
	
	
	
	/**
	 * Mark interval, which is an optional field, as non-existent.
	 */
	void clearInterval ();
	


	
	// ===> Attribute numAntenna, which is optional
	
	
	
	/**
	 * The attribute numAntenna is optional. Return true if this attribute exists.
	 * @return true if and only if the numAntenna attribute exists. 
	 */
	bool isNumAntennaExists() const;
	

	
 	/**
 	 * Get numAntenna, which is optional.
 	 * @return numAntenna as int
 	 * @throws IllegalAccessException If numAntenna does not exist.
 	 */
 	int getNumAntenna() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set numAntenna with the specified int.
 	 * @param numAntenna The int value to which numAntenna is to be set.
 	 
 		
 	 */
 	void setNumAntenna (int numAntenna);
		
	
	
	
	/**
	 * Mark numAntenna, which is an optional field, as non-existent.
	 */
	void clearNumAntenna ();
	


	
	// ===> Attribute numBaseband, which is optional
	
	
	
	/**
	 * The attribute numBaseband is optional. Return true if this attribute exists.
	 * @return true if and only if the numBaseband attribute exists. 
	 */
	bool isNumBasebandExists() const;
	

	
 	/**
 	 * Get numBaseband, which is optional.
 	 * @return numBaseband as int
 	 * @throws IllegalAccessException If numBaseband does not exist.
 	 */
 	int getNumBaseband() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set numBaseband with the specified int.
 	 * @param numBaseband The int value to which numBaseband is to be set.
 	 
 		
 	 */
 	void setNumBaseband (int numBaseband);
		
	
	
	
	/**
	 * Mark numBaseband, which is an optional field, as non-existent.
	 */
	void clearNumBaseband ();
	


	
	// ===> Attribute basebandName, which is optional
	
	
	
	/**
	 * The attribute basebandName is optional. Return true if this attribute exists.
	 * @return true if and only if the basebandName attribute exists. 
	 */
	bool isBasebandNameExists() const;
	

	
 	/**
 	 * Get basebandName, which is optional.
 	 * @return basebandName as vector<BasebandNameMod::BasebandName >
 	 * @throws IllegalAccessException If basebandName does not exist.
 	 */
 	vector<BasebandNameMod::BasebandName > getBasebandName() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set basebandName with the specified vector<BasebandNameMod::BasebandName >.
 	 * @param basebandName The vector<BasebandNameMod::BasebandName > value to which basebandName is to be set.
 	 
 		
 	 */
 	void setBasebandName (vector<BasebandNameMod::BasebandName > basebandName);
		
	
	
	
	/**
	 * Mark basebandName, which is an optional field, as non-existent.
	 */
	void clearBasebandName ();
	


	
	// ===> Attribute dValue, which is optional
	
	
	
	/**
	 * The attribute dValue is optional. Return true if this attribute exists.
	 * @return true if and only if the dValue attribute exists. 
	 */
	bool isDValueExists() const;
	

	
 	/**
 	 * Get dValue, which is optional.
 	 * @return dValue as double
 	 * @throws IllegalAccessException If dValue does not exist.
 	 */
 	double getDValue() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set dValue with the specified double.
 	 * @param dValue The double value to which dValue is to be set.
 	 
 		
 	 */
 	void setDValue (double dValue);
		
	
	
	
	/**
	 * Mark dValue, which is an optional field, as non-existent.
	 */
	void clearDValue ();
	


	
	// ===> Attribute vdValue, which is optional
	
	
	
	/**
	 * The attribute vdValue is optional. Return true if this attribute exists.
	 * @return true if and only if the vdValue attribute exists. 
	 */
	bool isVdValueExists() const;
	

	
 	/**
 	 * Get vdValue, which is optional.
 	 * @return vdValue as vector<double >
 	 * @throws IllegalAccessException If vdValue does not exist.
 	 */
 	vector<double > getVdValue() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set vdValue with the specified vector<double >.
 	 * @param vdValue The vector<double > value to which vdValue is to be set.
 	 
 		
 	 */
 	void setVdValue (vector<double > vdValue);
		
	
	
	
	/**
	 * Mark vdValue, which is an optional field, as non-existent.
	 */
	void clearVdValue ();
	


	
	// ===> Attribute vvdValues, which is optional
	
	
	
	/**
	 * The attribute vvdValues is optional. Return true if this attribute exists.
	 * @return true if and only if the vvdValues attribute exists. 
	 */
	bool isVvdValuesExists() const;
	

	
 	/**
 	 * Get vvdValues, which is optional.
 	 * @return vvdValues as vector<vector<double > >
 	 * @throws IllegalAccessException If vvdValues does not exist.
 	 */
 	vector<vector<double > > getVvdValues() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set vvdValues with the specified vector<vector<double > >.
 	 * @param vvdValues The vector<vector<double > > value to which vvdValues is to be set.
 	 
 		
 	 */
 	void setVvdValues (vector<vector<double > > vvdValues);
		
	
	
	
	/**
	 * Mark vvdValues, which is an optional field, as non-existent.
	 */
	void clearVvdValues ();
	


	
	// ===> Attribute llValue, which is optional
	
	
	
	/**
	 * The attribute llValue is optional. Return true if this attribute exists.
	 * @return true if and only if the llValue attribute exists. 
	 */
	bool isLlValueExists() const;
	

	
 	/**
 	 * Get llValue, which is optional.
 	 * @return llValue as long long
 	 * @throws IllegalAccessException If llValue does not exist.
 	 */
 	long long getLlValue() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set llValue with the specified long long.
 	 * @param llValue The long long value to which llValue is to be set.
 	 
 		
 	 */
 	void setLlValue (long long llValue);
		
	
	
	
	/**
	 * Mark llValue, which is an optional field, as non-existent.
	 */
	void clearLlValue ();
	


	
	// ===> Attribute vllValue, which is optional
	
	
	
	/**
	 * The attribute vllValue is optional. Return true if this attribute exists.
	 * @return true if and only if the vllValue attribute exists. 
	 */
	bool isVllValueExists() const;
	

	
 	/**
 	 * Get vllValue, which is optional.
 	 * @return vllValue as vector<long long >
 	 * @throws IllegalAccessException If vllValue does not exist.
 	 */
 	vector<long long > getVllValue() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set vllValue with the specified vector<long long >.
 	 * @param vllValue The vector<long long > value to which vllValue is to be set.
 	 
 		
 	 */
 	void setVllValue (vector<long long > vllValue);
		
	
	
	
	/**
	 * Mark vllValue, which is an optional field, as non-existent.
	 */
	void clearVllValue ();
	


	
	// ===> Attribute vvllValue, which is optional
	
	
	
	/**
	 * The attribute vvllValue is optional. Return true if this attribute exists.
	 * @return true if and only if the vvllValue attribute exists. 
	 */
	bool isVvllValueExists() const;
	

	
 	/**
 	 * Get vvllValue, which is optional.
 	 * @return vvllValue as vector<vector<long long > >
 	 * @throws IllegalAccessException If vvllValue does not exist.
 	 */
 	vector<vector<long long > > getVvllValue() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set vvllValue with the specified vector<vector<long long > >.
 	 * @param vvllValue The vector<vector<long long > > value to which vvllValue is to be set.
 	 
 		
 	 */
 	void setVvllValue (vector<vector<long long > > vvllValue);
		
	
	
	
	/**
	 * Mark vvllValue, which is an optional field, as non-existent.
	 */
	void clearVvllValue ();
	


	
	// ===> Attribute annotationId
	
	
	

	
 	/**
 	 * Get annotationId.
 	 * @return annotationId as Tag
 	 */
 	Tag getAnnotationId() const;
	
 
 	
 	
	
	


	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute antennaId, which is optional
	
	
	
	/**
	 * The attribute antennaId is optional. Return true if this attribute exists.
	 * @return true if and only if the antennaId attribute exists. 
	 */
	bool isAntennaIdExists() const;
	

	
 	/**
 	 * Get antennaId, which is optional.
 	 * @return antennaId as vector<Tag> 
 	 * @throws IllegalAccessException If antennaId does not exist.
 	 */
 	vector<Tag>  getAntennaId() const throw(IllegalAccessException);
	
 
 	
 	
 	/**
 	 * Set antennaId with the specified vector<Tag> .
 	 * @param antennaId The vector<Tag>  value to which antennaId is to be set.
 	 
 		
 	 */
 	void setAntennaId (vector<Tag>  antennaId);
		
	
	
	
	/**
	 * Mark antennaId, which is an optional field, as non-existent.
	 */
	void clearAntennaId ();
	


	///////////
	// Links //
	///////////
	
	
 		
 	/**
 	 * Set antennaId[i] with the specified Tag.
 	 * @param i The index in antennaId where to set the Tag value.
 	 * @param antennaId The Tag value to which antennaId[i] is to be set. 
 	 * @throws OutOfBoundsException
  	 */
  	void setAntennaId (int i, Tag antennaId)  ;
 			
	

	
		 
/**
 * Append a Tag to antennaId.
 * @param id the Tag to be appended to antennaId
 */
 void addAntennaId(Tag id); 

/**
 * Append a vector of Tag to antennaId.
 * @param id an array of Tag to be appended to antennaId
 */
 void addAntennaId(const vector<Tag> & id); 
 

 /**
  * Returns the Tag stored in antennaId at position i.
  * @param i the position in antennaId where the Tag is retrieved.
  * @return the Tag stored at position i in antennaId.
  */
 const Tag getAntennaId(int i);
 
 /**
  * Returns the AntennaRow linked to this row via the tag stored in antennaId
  * at position i.
  * @param i the position in antennaId.
  * @return a pointer on a AntennaRow whose key (a Tag) is equal to the Tag stored at position
  * i in the antennaId. 
  */
 AntennaRow* getAntenna(int i); 
 
 /**
  * Returns the vector of AntennaRow* linked to this row via the Tags stored in antennaId
  * @return an array of pointers on AntennaRow.
  */
 vector<AntennaRow *> getAntennas(); 
  

	

	
	
	
	/**
	 * Compare each mandatory attribute except the autoincrementable one of this AnnotationRow with 
	 * the corresponding parameters and return true if there is a match and false otherwise.
	 */ 
	bool compareNoAutoInc(ArrayTime time, string issue, string details);
	
	

	
	bool compareRequiredValue(ArrayTime time, string issue, string details); 
		 
	
	/**
	 * Return true if all required attributes of the value part are equal to their homologues
	 * in x and false otherwise.
	 *
	 * @param x a pointer on the AnnotationRow whose required attributes of the value part 
	 * will be compared with those of this.
	 * @return a boolean.
	 */
	bool equalByRequiredValue(AnnotationRow* x) ;

private:
	/**
	 * The table to which this row belongs.
	 */
	AnnotationTable &table;
	/**
	 * Whether this row has been added to the table or not.
	 */
	bool hasBeenAdded;

	// This method is used by the Table class when this row is added to the table.
	void isAdded();


	/**
	 * Create a AnnotationRow.
	 * <p>
	 * This constructor is private because only the
	 * table can create rows.  All rows know the table
	 * to which they belong.
	 * @param table The table to which this row belongs.
	 */ 
	AnnotationRow (AnnotationTable &table);

	/**
	 * Create a AnnotationRow using a copy constructor mechanism.
	 * <p>
	 * Given a AnnotationRow row and a AnnotationTable table, the method creates a new
	 * AnnotationRow owned by table. Each attribute of the created row is a copy (deep)
	 * of the corresponding attribute of row. The method does not add the created
	 * row to its table, its simply parents it to table, a call to the add method
	 * has to be done in order to get the row added (very likely after having modified
	 * some of its attributes).
	 * If row is null then the method returns a row with default values for its attributes. 
	 *
	 * This constructor is private because only the
	 * table can create rows.  All rows know the table
	 * to which they belong.
	 * @param table The table to which this row belongs.
	 * @param row  The row which is to be copied.
	 */
	 AnnotationRow (AnnotationTable &table, AnnotationRow &row);
	 	
	////////////////////////////////
	// Intrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute time
	
	

	ArrayTime time;

	
	
 	

	
	// ===> Attribute issue
	
	

	string issue;

	
	
 	

	
	// ===> Attribute details
	
	

	string details;

	
	
 	

	
	// ===> Attribute interval, which is optional
	
	
	bool intervalExists;
	

	Interval interval;

	
	
 	

	
	// ===> Attribute numAntenna, which is optional
	
	
	bool numAntennaExists;
	

	int numAntenna;

	
	
 	

	
	// ===> Attribute numBaseband, which is optional
	
	
	bool numBasebandExists;
	

	int numBaseband;

	
	
 	

	
	// ===> Attribute basebandName, which is optional
	
	
	bool basebandNameExists;
	

	vector<BasebandNameMod::BasebandName > basebandName;

	
	
 	

	
	// ===> Attribute dValue, which is optional
	
	
	bool dValueExists;
	

	double dValue;

	
	
 	

	
	// ===> Attribute vdValue, which is optional
	
	
	bool vdValueExists;
	

	vector<double > vdValue;

	
	
 	

	
	// ===> Attribute vvdValues, which is optional
	
	
	bool vvdValuesExists;
	

	vector<vector<double > > vvdValues;

	
	
 	

	
	// ===> Attribute llValue, which is optional
	
	
	bool llValueExists;
	

	long long llValue;

	
	
 	

	
	// ===> Attribute vllValue, which is optional
	
	
	bool vllValueExists;
	

	vector<long long > vllValue;

	
	
 	

	
	// ===> Attribute vvllValue, which is optional
	
	
	bool vvllValueExists;
	

	vector<vector<long long > > vvllValue;

	
	
 	

	
	// ===> Attribute annotationId
	
	

	Tag annotationId;

	
	
 	
 	/**
 	 * Set annotationId with the specified Tag value.
 	 * @param annotationId The Tag value to which annotationId is to be set.
		
 		
			
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 		
 	 */
 	void setAnnotationId (Tag annotationId);
  		
	

	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	
	// ===> Attribute antennaId, which is optional
	
	
	bool antennaIdExists;
	

	vector<Tag>  antennaId;

	
	
 	

	///////////
	// Links //
	///////////
	
	
		


	


};

} // End namespace asdm

#endif /* Annotation_CLASS */
