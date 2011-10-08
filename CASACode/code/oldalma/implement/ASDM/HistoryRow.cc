
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
 * File HistoryRow.cpp
 */
 
#include <vector>
using std::vector;

#include <set>
using std::set;

#include <ASDM.h>
#include <HistoryRow.h>
#include <HistoryTable.h>

#include <ExecBlockTable.h>
#include <ExecBlockRow.h>
	

using asdm::ASDM;
using asdm::HistoryRow;
using asdm::HistoryTable;

using asdm::ExecBlockTable;
using asdm::ExecBlockRow;


#include <Parser.h>
using asdm::Parser;

#include <EnumerationParser.h>
 
#include <InvalidArgumentException.h>
using asdm::InvalidArgumentException;

namespace asdm {

	HistoryRow::~HistoryRow() {
	}

	/**
	 * Return the table to which this row belongs.
	 */
	HistoryTable &HistoryRow::getTable() const {
		return table;
	}
	
	void HistoryRow::isAdded() {
		hasBeenAdded = true;
	}
	
	
#ifndef WITHOUT_ACS
	/**
	 * Return this row in the form of an IDL struct.
	 * @return The values of this row as a HistoryRowIDL struct.
	 */
	HistoryRowIDL *HistoryRow::toIDL() const {
		HistoryRowIDL *x = new HistoryRowIDL ();
		
		// Fill the IDL structure.
	
		
	
  		
		
		
			
		x->time = time.toIDLArrayTime();
			
		
	

	
  		
		
		
			
				
		x->message = CORBA::string_dup(message.c_str());
				
 			
		
	

	
  		
		
		
			
				
		x->priority = CORBA::string_dup(priority.c_str());
				
 			
		
	

	
  		
		
		
			
				
		x->origin = CORBA::string_dup(origin.c_str());
				
 			
		
	

	
  		
		
		
			
				
		x->objectId = CORBA::string_dup(objectId.c_str());
				
 			
		
	

	
  		
		
		
			
				
		x->application = CORBA::string_dup(application.c_str());
				
 			
		
	

	
  		
		
		
			
				
		x->cliCommand = CORBA::string_dup(cliCommand.c_str());
				
 			
		
	

	
  		
		
		
			
				
		x->appParms = CORBA::string_dup(appParms.c_str());
				
 			
		
	

	
	
		
	
  	
 		
		
	 	
			
		x->execBlockId = execBlockId.toIDLTag();
			
	 	 		
  	

	
		
	

		
		return x;
	
	}
#endif
	

#ifndef WITHOUT_ACS
	/**
	 * Fill the values of this row from the IDL struct HistoryRowIDL.
	 * @param x The IDL struct containing the values used to fill this row.
	 */
	void HistoryRow::setFromIDL (HistoryRowIDL x) throw(ConversionException) {
		try {
		// Fill the values from x.
	
		
	
		
		
			
		setTime(ArrayTime (x.time));
			
 		
		
	

	
		
		
			
		setMessage(string (x.message));
			
 		
		
	

	
		
		
			
		setPriority(string (x.priority));
			
 		
		
	

	
		
		
			
		setOrigin(string (x.origin));
			
 		
		
	

	
		
		
			
		setObjectId(string (x.objectId));
			
 		
		
	

	
		
		
			
		setApplication(string (x.application));
			
 		
		
	

	
		
		
			
		setCliCommand(string (x.cliCommand));
			
 		
		
	

	
		
		
			
		setAppParms(string (x.appParms));
			
 		
		
	

	
	
		
	
		
		
			
		setExecBlockId(Tag (x.execBlockId));
			
 		
		
	

	
		
	

		} catch (IllegalAccessException err) {
			throw new ConversionException (err.getMessage(),"History");
		}
	}
#endif
	
	/**
	 * Return this row in the form of an XML string.
	 * @return The values of this row as an XML string.
	 */
	string HistoryRow::toXML() const {
		string buf;
		buf.append("<row> \n");
		
	
		
  	
 		
		
		Parser::toXML(time, "time", buf);
		
		
	

  	
 		
		
		Parser::toXML(message, "message", buf);
		
		
	

  	
 		
		
		Parser::toXML(priority, "priority", buf);
		
		
	

  	
 		
		
		Parser::toXML(origin, "origin", buf);
		
		
	

  	
 		
		
		Parser::toXML(objectId, "objectId", buf);
		
		
	

  	
 		
		
		Parser::toXML(application, "application", buf);
		
		
	

  	
 		
		
		Parser::toXML(cliCommand, "cliCommand", buf);
		
		
	

  	
 		
		
		Parser::toXML(appParms, "appParms", buf);
		
		
	

	
	
		
  	
 		
		
		Parser::toXML(execBlockId, "execBlockId", buf);
		
		
	

	
		
	

		
		buf.append("</row>\n");
		return buf;
	}

	/**
	 * Fill the values of this row from an XML string 
	 * that was produced by the toXML() method.
	 * @param x The XML string being used to set the values of this row.
	 */
	void HistoryRow::setFromXML (string rowDoc) throw(ConversionException) {
		Parser row(rowDoc);
		string s = "";
		try {
	
		
	
  		
			
	  	setTime(Parser::getArrayTime("time","History",rowDoc));
			
		
	

	
  		
			
	  	setMessage(Parser::getString("message","History",rowDoc));
			
		
	

	
  		
			
	  	setPriority(Parser::getString("priority","History",rowDoc));
			
		
	

	
  		
			
	  	setOrigin(Parser::getString("origin","History",rowDoc));
			
		
	

	
  		
			
	  	setObjectId(Parser::getString("objectId","History",rowDoc));
			
		
	

	
  		
			
	  	setApplication(Parser::getString("application","History",rowDoc));
			
		
	

	
  		
			
	  	setCliCommand(Parser::getString("cliCommand","History",rowDoc));
			
		
	

	
  		
			
	  	setAppParms(Parser::getString("appParms","History",rowDoc));
			
		
	

	
	
		
	
  		
			
	  	setExecBlockId(Parser::getTag("execBlockId","ExecBlock",rowDoc));
			
		
	

	
		
	

		} catch (IllegalAccessException err) {
			throw ConversionException (err.getMessage(),"History");
		}
	}
	
	////////////////////////////////
	// Intrinsic Table Attributes //
	////////////////////////////////
	
	

	
 	/**
 	 * Get time.
 	 * @return time as ArrayTime
 	 */
 	ArrayTime HistoryRow::getTime() const {
	
  		return time;
 	}

 	/**
 	 * Set time with the specified ArrayTime.
 	 * @param time The ArrayTime value to which time is to be set.
 	 
 	
 		
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 	
 	 */
 	void HistoryRow::setTime (ArrayTime time)  {
  	
  	
  		if (hasBeenAdded) {
 		
			throw IllegalAccessException("time", "History");
		
  		}
  	
 		this->time = time;
	
 	}
	
	

	

	
 	/**
 	 * Get message.
 	 * @return message as string
 	 */
 	string HistoryRow::getMessage() const {
	
  		return message;
 	}

 	/**
 	 * Set message with the specified string.
 	 * @param message The string value to which message is to be set.
 	 
 	
 		
 	 */
 	void HistoryRow::setMessage (string message)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->message = message;
	
 	}
	
	

	

	
 	/**
 	 * Get priority.
 	 * @return priority as string
 	 */
 	string HistoryRow::getPriority() const {
	
  		return priority;
 	}

 	/**
 	 * Set priority with the specified string.
 	 * @param priority The string value to which priority is to be set.
 	 
 	
 		
 	 */
 	void HistoryRow::setPriority (string priority)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->priority = priority;
	
 	}
	
	

	

	
 	/**
 	 * Get origin.
 	 * @return origin as string
 	 */
 	string HistoryRow::getOrigin() const {
	
  		return origin;
 	}

 	/**
 	 * Set origin with the specified string.
 	 * @param origin The string value to which origin is to be set.
 	 
 	
 		
 	 */
 	void HistoryRow::setOrigin (string origin)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->origin = origin;
	
 	}
	
	

	

	
 	/**
 	 * Get objectId.
 	 * @return objectId as string
 	 */
 	string HistoryRow::getObjectId() const {
	
  		return objectId;
 	}

 	/**
 	 * Set objectId with the specified string.
 	 * @param objectId The string value to which objectId is to be set.
 	 
 	
 		
 	 */
 	void HistoryRow::setObjectId (string objectId)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->objectId = objectId;
	
 	}
	
	

	

	
 	/**
 	 * Get application.
 	 * @return application as string
 	 */
 	string HistoryRow::getApplication() const {
	
  		return application;
 	}

 	/**
 	 * Set application with the specified string.
 	 * @param application The string value to which application is to be set.
 	 
 	
 		
 	 */
 	void HistoryRow::setApplication (string application)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->application = application;
	
 	}
	
	

	

	
 	/**
 	 * Get cliCommand.
 	 * @return cliCommand as string
 	 */
 	string HistoryRow::getCliCommand() const {
	
  		return cliCommand;
 	}

 	/**
 	 * Set cliCommand with the specified string.
 	 * @param cliCommand The string value to which cliCommand is to be set.
 	 
 	
 		
 	 */
 	void HistoryRow::setCliCommand (string cliCommand)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->cliCommand = cliCommand;
	
 	}
	
	

	

	
 	/**
 	 * Get appParms.
 	 * @return appParms as string
 	 */
 	string HistoryRow::getAppParms() const {
	
  		return appParms;
 	}

 	/**
 	 * Set appParms with the specified string.
 	 * @param appParms The string value to which appParms is to be set.
 	 
 	
 		
 	 */
 	void HistoryRow::setAppParms (string appParms)  {
  	
  	
  		if (hasBeenAdded) {
 		
  		}
  	
 		this->appParms = appParms;
	
 	}
	
	

	
	////////////////////////////////
	// Extrinsic Table Attributes //
	////////////////////////////////
	
	

	
 	/**
 	 * Get execBlockId.
 	 * @return execBlockId as Tag
 	 */
 	Tag HistoryRow::getExecBlockId() const {
	
  		return execBlockId;
 	}

 	/**
 	 * Set execBlockId with the specified Tag.
 	 * @param execBlockId The Tag value to which execBlockId is to be set.
 	 
 	
 		
 	 * @throw IllegalAccessException If an attempt is made to change this field after is has been added to the table.
 	 	
 	 */
 	void HistoryRow::setExecBlockId (Tag execBlockId)  {
  	
  	
  		if (hasBeenAdded) {
 		
			throw IllegalAccessException("execBlockId", "History");
		
  		}
  	
 		this->execBlockId = execBlockId;
	
 	}
	
	

	///////////
	// Links //
	///////////
	
	
	
	
		

	/**
	 * Returns the pointer to the row in the ExecBlock table having ExecBlock.execBlockId == execBlockId
	 * @return a ExecBlockRow*
	 * 
	 
	 */
	 ExecBlockRow* HistoryRow::getExecBlockUsingExecBlockId() {
	 
	 	return table.getContainer().getExecBlock().getRowByKey(execBlockId);
	 }
	 

	

	
	/**
	 * Create a HistoryRow.
	 * <p>
	 * This constructor is private because only the
	 * table can create rows.  All rows know the table
	 * to which they belong.
	 * @param table The table to which this row belongs.
	 */ 
	HistoryRow::HistoryRow (HistoryTable &t) : table(t) {
		hasBeenAdded = false;
		
	
	

	

	

	

	

	

	

	

	
	

	
	
	
	

	

	

	

	

	

	

	
	
	}
	
	HistoryRow::HistoryRow (HistoryTable &t, HistoryRow &row) : table(t) {
		hasBeenAdded = false;
		
		if (&row == 0) {
	
	
	

	

	

	

	

	

	

	

	
	
		
		}
		else {
	
		
			execBlockId = row.execBlockId;
		
			time = row.time;
		
		
		
		
			message = row.message;
		
			priority = row.priority;
		
			origin = row.origin;
		
			objectId = row.objectId;
		
			application = row.application;
		
			cliCommand = row.cliCommand;
		
			appParms = row.appParms;
		
		
		
		
		}	
	}

	
	bool HistoryRow::compareNoAutoInc(Tag execBlockId, ArrayTime time, string message, string priority, string origin, string objectId, string application, string cliCommand, string appParms) {
		bool result;
		result = true;
		
	
		
		result = result && (this->execBlockId == execBlockId);
		
		if (!result) return false;
	

	
		
		result = result && (this->time == time);
		
		if (!result) return false;
	

	
		
		result = result && (this->message == message);
		
		if (!result) return false;
	

	
		
		result = result && (this->priority == priority);
		
		if (!result) return false;
	

	
		
		result = result && (this->origin == origin);
		
		if (!result) return false;
	

	
		
		result = result && (this->objectId == objectId);
		
		if (!result) return false;
	

	
		
		result = result && (this->application == application);
		
		if (!result) return false;
	

	
		
		result = result && (this->cliCommand == cliCommand);
		
		if (!result) return false;
	

	
		
		result = result && (this->appParms == appParms);
		
		if (!result) return false;
	

		return result;
	}	
	
	
	
	bool HistoryRow::compareRequiredValue(string message, string priority, string origin, string objectId, string application, string cliCommand, string appParms) {
		bool result;
		result = true;
		
	
		if (!(this->message == message)) return false;
	

	
		if (!(this->priority == priority)) return false;
	

	
		if (!(this->origin == origin)) return false;
	

	
		if (!(this->objectId == objectId)) return false;
	

	
		if (!(this->application == application)) return false;
	

	
		if (!(this->cliCommand == cliCommand)) return false;
	

	
		if (!(this->appParms == appParms)) return false;
	

		return result;
	}
	
	
	/**
	 * Return true if all required attributes of the value part are equal to their homologues
	 * in x and false otherwise.
	 *
	 * @param x a pointer on the HistoryRow whose required attributes of the value part 
	 * will be compared with those of this.
	 * @return a boolean.
	 */
	bool HistoryRow::equalByRequiredValue(HistoryRow* x) {
		
			
		if (this->message != x->message) return false;
			
		if (this->priority != x->priority) return false;
			
		if (this->origin != x->origin) return false;
			
		if (this->objectId != x->objectId) return false;
			
		if (this->application != x->application) return false;
			
		if (this->cliCommand != x->cliCommand) return false;
			
		if (this->appParms != x->appParms) return false;
			
		
		return true;
	}	
	

} // End namespace asdm
 