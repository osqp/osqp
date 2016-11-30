% OSQP : Matlab class wrapper for OSQP solver
classdef osqp < handle
    properties (SetAccess = private, Hidden = true)
        objectHandle; % Handle to underlying C instance
    end
    methods
        %% Constructor - Create a new C instance
        function this = osqp(varargin)
            this.objectHandle = osqp_mex('new', varargin{:});
        end

        %% Destructor - Destroy the C instance
        function delete(this)
            osqp_mex('delete', this.objectHandle);
        end

        %% setup : configure solver with problem data
        %%
        %% Usage : setup(osqpHandle,P,q,A,lA,uA)
        function varargout = setup(this, varargin)
            
            %dimension checks on user data. Mex function does not
            %perform any checks on inputs, so check everything here
            try
                assert(length(varargin) == 5);
                [P,q,A,lA,uA] = deal(varargin{:});
                P   = sparse(P);
                q   = full(q(:));
                A   = sparse(A);
                lA  = full(lA(:));
                uA  = full(uA(:));
                assert(size(P,1)  == size(P,2)); 
                assert(size(P,2)  == size(A,2));
                assert(length(q)  == size(A,1));
                assert(length(lA) == size(A,1));
                assert(length(uA) == size(A,1));
            catch
                error('Incorrect number or type of input arguments');
            end
            [varargout{1:nargout}] = osqp_mex('setup', this.objectHandle, P,q,A,lA,uA);
        end

        %% Test - another example class method call
        function varargout = solve(this, varargin)
            [varargout{1:nargout}] =osqp_mex('solve', this.objectHandle);
        end
    end
end
