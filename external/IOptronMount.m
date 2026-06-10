classdef IOptronMount <handle
    
    properties % Position
        Az
        Alt
        Dec
        RA
    end
        
    properties(GetAccess=public, SetAccess=private)
        Status
    end

    properties(Hidden) % interrogable, but not of immediate use
        Port
        AltUserLimit
        ParkPosition
        Time
        verbose=true;
    end
 
    methods % Constructor and communication commands
        
        function I=IOptronMount(port) % Constructor
            % Constructor, connect the serial port
            if ~exist('port','var')
                port='';
            end
            I.SetPort(port);
            % check if the mount is there by querying something
            try
                model=I.Query('MountInfo');
%                I.Az=[]; % temporary, remove
                if ~strcmp(model(1:3),'012')
                    I.report(['no IOptron mount found on ',port,'\n'])
                end
            catch
                I.report(['no IOptron mount found on ',port,'\n'])
            end
        end
        
        function I=SetPort(I,port)
            if ~exist('port','var') || isempty(port)
                for port=seriallist
                    try
                        % look for one IOptron device on every
                        %  possible serial port. Pity we cannot
                        % look for a named (i.e. SN) unit
                        I.SetPort(port);
                        if ~(isempty(I.Az))
                            I.report('An IOptron mount was found on '+port+'\n')
                            break
                        else
                            I.report('no IOptron mount found on '+port+'\n')
                        end
                    catch
                        I.report('no IOptron mount found on '+port+'\n')
                    end
                end
            end
            try
                delete(instrfind('Port',port))
            catch
            end
            try
                I.Port=serial(port);
                % serial has been deprecated in 2019b in favour of
                %  serialport... all communication code should be
                %  transitioned...
            catch
            end
            try
                if strcmp(I.Port.status,'closed')
                    fopen(I.Port);
                end
            catch
                error(['Port ' I.Port.name ' cannot be opened'])
            end
            set(I.Port,'BaudRate',115200,'Timeout',1);
        end
        
        function resp=Query(I,cmd)
            % Dispose of previous traffic potentially having
            % filled the inbuffer, for an immediate response
            flushinput(I.Port)
            fprintf(I.Port,':%s#',cmd);
            if strcmp(cmd,'Q')
                pause(0.5); % abort requires a longer delay
            elseif strcmp(cmd(1:2),'ST')
                pause(0.7); % start and stop tracking even longer
            else
                pause(0.1);
            end
            resp=char(fread(I.Port,[1,I.Port.BytesAvailable],'char'));
            % possible replies are long strings terminated by #
            %  for get commands, or 0/1 for boolean gets, or setters
            if isempty(resp)
                error('Mount did not respond. Maybe wrong command?')
            end
            if ~strcmp(resp(end),'#') && ...
                   (numel(resp)==1 && ~(resp=='0' || resp=='1'))
                error('Response from mount incomplete')
            end
        end

        
        function Close(I)
            fclose(I.Port);
        end
        
        function delete(I)
            delete(I.Port)
        end
        
    end
    
    methods % getter/setters: Position and status
        
        function S=get.Status(I)
            % state enumerations - let the function error if an out-of
            %  -range value is returned
            gpsstate=["No GPS","no data","valid"];
            motionstate=["stopped","track without PEC","slew","auto-guiding",...
                         "meridian flipping","track with PEC","parked",...
                         "at home"];
            trackingrate=["sidereal","lunar","solar","King","custom"];
            keyspeed=["1x","2x","4x","8x","16x","32x","64x","128x",...
                      "256x","512x","max"];
            timesource=["communicated","hand controller","GPS"];
            hemisphere=["South","North"];
            resp=I.Query('GLS');
            S=struct('Lon',str2double(resp(1:9))/360000,...
                     'Lat',str2double(resp(10:17))/360000-90,...
                     'GPS',gpsstate(str2double(resp(18))+1),...
                     'motion',motionstate(str2double(resp(19))+1),...
                     'tracking',trackingrate(str2double(resp(20))+1),...
                     'keyspeed',keyspeed(str2double(resp(21))+1),...
                     'timesource',timesource(str2double(resp(22))+1),...
                     'hempisphere',hemisphere(str2double(resp(23))+1) );
        end

        function AZ=get.Az(I)
            resp=I.Query('GAC');
            AZ=str2double(resp(10:18))/360000;
        end
        
        function set.Az(I,AZ)
            I.Query(sprintf('Sz%09d',int32(AZ*360000)));
            resp=I.Query('MSS');
            if resp~='1'
                error('target position beyond limits')
            end
        end
        
        function ALT=get.Alt(I)
            resp=I.Query('GAC');
            ALT=str2double(resp(1:9))/360000;
        end
        
        function set.Alt(I,ALT)
            I.Query(sprintf('Sa%+09d',int32(ALT*360000)));
            resp=I.Query('MSS');
            if resp~='1'
                error('target position beyond limits')
            end
        end
        
        function DEC=get.Dec(I)
            resp=I.Query('GEP');
            DEC=str2double(resp(1:9))/360000;
        end
        
        function set.Dec(I,DEC)
            I.Query(sprintf('Sd%+08d',int32(DEC*360000)));
            resp=I.Query('MS1');
            if resp~='1'
                error('target position beyond limits')
            end
        end
        
        function RA=get.RA(I)
            resp=I.Query('GEP');
            RA=str2double(resp(10:18))/360000;
        end
        
        % East or West of pier, and counterweight positions could
        %  be read from the last two digits of the answer to GEP.
        %  However, they should also be understandable from Az and Alt (?)
 
        function set.RA(I,RA)
            I.Query(sprintf('SRA%09d',int32(RA*360000)));
            resp=I.Query('MS1'); % choose counterweight down for now
            if resp~='1'
                error('target position beyond limits')
            end
        end
        
        function T=get.Time(I)
            resp=I.Query('GUT');
            T.UTC_offset=str2double(resp(1:4));
            T.DST=(resp(5)=='1');
            % UTC time in secs, = (JD-J2000)*3600*24
            %  actulally given in ms, but the resolution is 1000ms
            T.UTC=str2double(resp(6:18))/1000;
            % in matlab form
            % T.datenum=T.UTC/24/3600+datenum('1/1/2000 12:00');
        end
        
        % setters for Time and Lon, Lat are needed if GPS is off
        
        function set.Time(I,T)
            % T structure with T.UTC_offset, T.DST, T.UTC
            if T.UTC_offset>=-720 && T.UTC_offset<=780
                I.Query(sprintf('SG%+03d',int16(T.UTC_offset)));
            else
                error('T.UTC_offset out of range')
            end
            if T.DST
                I.Query('SDS1');
            else
                I.Query('SDS0');
            end
            % T.UTC = (T.datenum-datenum('1/1/2000 12:00'))*24*3600
            if T.UTC>0
                I.Query(sprintf('SUT%013d',int32(T.UTC*1000)));
            else
                error('T.UTC must be greater than 0, t>J2000')
            end
        end
        
        function setLonLat(I,Lon,Lat,hem)
            I.Query(sprintf('SLO%+09d',int32(Lon*360000)));
            I.Query(sprintf('SLA%+09d',int32(Lat*360000)));
            if hem
                % 1 is north
                I.Query('SHE1');
            else
                % 0 is south
                I.Query('SHE0');
            end
        end
        
    end
    
    methods % Moving commands.
        
        function Abort(I)
            % emergency stop
            I.Query('Q');
            I.Query('ST0');
        end
        
        function GoHome(I)
            I.Query('MH');
        end
        
        function FullHoming(I)
            I.Query('MSH');
            % here, poll status and exit only when done
            I.report('searching home')
            retry=0; maxretry=100;
            while ~strcmp(I.Status.motion,'at home') && retry<maxretry
                pause(.5)
                retry=retry+1;
                I.report('.')
            end
            if retry<maxretry
                I.report(' done!\n')
            else
                I.report('\n')
                error('homing not attained in due time')
            end
        end
        
        function flag=isHomed(I)
            % a bit redundant, to duplicate the same function of NexStarPCport
            flag=strcmp(I.Status.motion,'at home');
        end

    end
    
    methods % functioning parameters getters/setters & misc
        
        function alt=get.AltUserLimit(I)
            resp=I.Query('GAL');
            alt=str2double(resp(1:3));
        end
        
        function set.AltUserLimit(I,alt)
            if alt<-89 || alt>89
                error('altitude limit illegal')
            else
                I.Query(sprintf('SAL%+03d',round(alt)));
            end
        end
       
        function p=get.ParkPosition(I)
            resp=I.Query('GPC');
            p.alt=str2double(resp(1:8))/360000;
            p.az=str2double(resp(9:17))/360000;
        end

        function set.ParkPosition(I,pos)
            % allow for convenience pos to be either a struct or an array
            if isstruct(pos)
                p=pos;
            else
                p.az=pos(1);
                p.alt=pos(2);
            end
            I.Query(sprintf('SPH%08d',int32(p.alt*360000)));
            I.Query(sprintf('SPA%08d',int32(p.az*360000)));
        end
        
        function Park(I)
            resp=I.Query('MP1');
            if resp~='1'
                error('parking mount failed')
            end
        end
        
        function Unpark(I)
            resp=I.Query('MP0');
            if resp~='1'
                error('unparking mount failed')
            end
        end
        
    end
    
    methods %slewing and tracking
        
        function Track(I,rate)
            % rate is either "sidereal","lunar","solar","King",
            %  or a number in the range 0.1:1.9 for "custom"
            %  rate in sidereal units. Rate=0 stops tracking
            if isnumeric(rate)
                if rate==0
                    I.Query('ST0');
                elseif rate>=0.1 && rate <=1.9
                    I.Query('RT4');
                    I.Query(sprintf('RR%05d',int32(rate*10000)));
                    I.Query('ST1');
                else
                    error('illegal tracking rate - should be [0.1:1.9] or 0 to stop')
                end
            else
                switch rate
                    case 'sidereal'
                        I.Query('RT0');
                    case 'lunar'
                        I.Query('RT1');
                    case 'solar'
                        I.Query('RT2');
                    case 'King'
                        I.Query('RT3');
                    otherwise
                        error('illegal rate - should be sidereal/solar/lunar/King')
                end
            end
        end
        
    end
    
    methods(Access=private)
        
        % verbose reporting
        function report(I,msg)
            if I.verbose
                fprintf(msg)
            end
        end
        
    end
end