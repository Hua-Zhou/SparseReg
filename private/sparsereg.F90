      MODULE SPARSEREG
!
!     Determine double precision and set constants.
!
      IMPLICIT NONE
      INTEGER , PARAMETER :: DBLE_PREC = KIND (0.0D0)
      REAL (KIND=DBLE_PREC), PARAMETER :: ZERO  = 0.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: ONE   = 1.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: TWO   = 2.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: THREE = 3.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: FOUR  = 4.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: FIVE  = 5.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: SIX   = 6.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: SEVEN = 7.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: EIGHT = 8.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: NINE  = 9.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: TEN   = 10.0_DBLE_PREC
      REAL (KIND=DBLE_PREC), PARAMETER :: HALF  = ONE/TWO
      REAL (KIND=DBLE_PREC), PARAMETER :: PI    = 3.14159265358979_DBLE_PREC
!
      CONTAINS
!
      SUBROUTINE READ_DATA(INPUT_FILE,X,Y,ESTIMATE,N,P)
!
!     THIS SUBROUTINE READS IN THE DATA AND INITIALIZES CONSTANTS AND ARRAYS.
!
      IMPLICIT NONE
      CHARACTER(LEN=100) :: INPUT_FILE
      INTEGER :: I,INPUT_UNIT=1,N,P
      REAL(KIND=DBLE_PREC), DIMENSION(:) :: ESTIMATE,Y
      REAL(KIND=DBLE_PREC), DIMENSION(:,:) :: X
!
!     Open the input file.
!
      OPEN(UNIT=INPUT_UNIT,FILE=INPUT_FILE)
!
!     Read the exponents
!
      READ(INPUT_UNIT,*) ESTIMATE(1:P)
      DO I = 1,N
	      READ(INPUT_UNIT,*) Y(I),X(I,1:P)
      END DO
      CLOSE(INPUT_UNIT)
      END SUBROUTINE READ_DATA      
!
      SUBROUTINE PENALTY_FUN(BETA,RHO,ETA,PENTYPE,PEN,D1PEN,D2PEN,DPENDRHO)
!
!     This subroutine calculates the penalty value, first derivative, 
!     and second derivatives of the ENET, LOG, MCP, POWER, or SCAD penalties
!     with index parameter ETA
!       ENET: rho*((eta-1)*beta**2/2+(2-eta)*abs(beta)), 1<=eta<=2
!       LOG: rho*log(eta+abs(beta)), eta>0 (eta=0 implies continuous LOG penalty)
!       MCP: eta>0
!       POWER: rho*abs(beta)**eta, 0<eta<=2
!       SCAD: eta>2
!
      IMPLICIT NONE
      CHARACTER(LEN=*), INTENT(IN) :: PENTYPE
      REAL(KIND=DBLE_PREC), INTENT(IN) :: ETA,RHO
      REAL(KIND=DBLE_PREC), INTENT(IN), DIMENSION(:) :: BETA
      REAL(KIND=DBLE_PREC), INTENT(OUT), DIMENSION(:) :: PEN
      REAL(KIND=DBLE_PREC), INTENT(OUT), OPTIONAL, DIMENSION(:) :: D1PEN,D2PEN,DPENDRHO
!
!     Local variables
!   
      LOGICAL :: CONTLOG
      REAL(KIND=DBLE_PREC), PARAMETER :: EPS=1E-8         
      REAL(KIND=DBLE_PREC) :: ETAC
      REAL(KIND=DBLE_PREC), DIMENSION(SIZE(BETA)) :: ABSBETA
!
!     Check nonnegativity of tuning parameter
!
      IF (RHO<ZERO) THEN
         !PRINT*,"THE TUNING PARAMETER MUST BE NONNEGATIVE."
         RETURN
      END IF
!
!     Penalty values
!
      ABSBETA = ABS(BETA)
      SELECT CASE(PENTYPE)
      CASE("ENET")
         IF (ETA<ONE.OR.ETA>TWO) THEN
            !PRINT*,"THE ENET PARAMETER ETA SHOULD BE IN [1,2]."
            RETURN
         END IF
         PEN = RHO*(HALF*(ETA-ONE)*BETA*BETA+(TWO-ETA)*ABSBETA)
      CASE("LOG")
         ETAC = ETA
         IF (ETA==ZERO) THEN
            ETAC = SQRT(RHO)
            CONTLOG = .TRUE.
         ELSEIF (ETA<ZERO) THEN
            !PRINT*,"THE LOG PENALTY PARAMETER ETA SHOULD BE NONNEGATIVE."
            RETURN
         END IF
         PEN = RHO*LOG(ETAC+ABSBETA)      
      CASE("MCP")
         IF (ETA<=ZERO) THEN
            !PRINT*,"THE MCP PARAMETER ETA SHOULD BE POSITIVE."
            RETURN
         END IF
         WHERE (ABSBETA<RHO*ETA)
            PEN = RHO*ABSBETA - HALF*BETA*BETA/ETA
         ELSEWHERE
            PEN = HALF*RHO*RHO*ETA
         END WHERE
      CASE("POWER")
         IF (ETA<=ZERO.OR.ETA>TWO) THEN
            !PRINT*,"THE EXPONENT PARAMETER ETA SHOULD BE IN (0,2]."
            RETURN
         END IF
         PEN = RHO*ABSBETA**ETA
      CASE("SCAD")
         IF (ETA<=TWO) THEN
            !PRINT*,"THE SCAD PARAMETER ETA SHOULD BE GREATER THAN 2."
            RETURN
         END IF
         WHERE (ABSBETA<RHO)
            PEN = RHO*ABSBETA
         ELSEWHERE (ABSBETA<ETA*RHO)
            PEN = RHO*RHO + (ETA*RHO*(ABSBETA-RHO) &
               - HALF*(BETA*BETA-RHO*RHO))/(ETA-ONE)
         ELSEWHERE
            PEN = HALF*RHO*RHO*(ETA+ONE)
         END WHERE
      END SELECT
!
!     First derivative of penalty function
!
      IF (PRESENT(D1PEN)) THEN
         SELECT CASE(PENTYPE)
         CASE("ENET")
            D1PEN = RHO*((ETA-ONE)*ABSBETA+TWO-ETA)
         CASE("LOG")
            D1PEN = RHO/(ETAC+ABSBETA)
         CASE("MCP")
            WHERE (ABSBETA<RHO*ETA)
               D1PEN = RHO - ABSBETA/ETA
            ELSEWHERE
               D1PEN = ZERO
            END WHERE
         CASE("POWER")
            WHERE (ABSBETA<EPS)
               D1PEN = RHO*ETA*EPS**(ETA-ONE)
            ELSEWHERE
               D1PEN = RHO*ETA*ABSBETA**(ETA-ONE)
            END WHERE               
         CASE("SCAD")
            WHERE (ABSBETA<RHO)
               D1PEN = RHO
            ELSEWHERE (ABSBETA<ETA*RHO)
               D1PEN = (ETA*RHO-ABSBETA)/(ETA-ONE)
            ELSEWHERE
               D1PEN = ZERO
            END WHERE         
         END SELECT
      END IF
!
!     Second derivative of penalty function
!
      IF (PRESENT(D2PEN)) THEN
         SELECT CASE(PENTYPE)
         CASE("ENET")
            D2PEN = RHO*(ETA-ONE)
         CASE("LOG")
            D2PEN = -D1PEN/(ETAC+ABSBETA)
         CASE("MCP")
            WHERE (ABSBETA<RHO*ETA)
               D2PEN = -ONE/ETA
            ELSEWHERE
               D2PEN = ZERO
            END WHERE
         CASE("POWER")
            WHERE (ABSBETA<EPS)
               D2PEN = RHO*ETA*(ETA-ONE)*EPS**(ETA-TWO)
            ELSEWHERE
               D2PEN = RHO*ETA*(ETA-ONE)*ABSBETA**(ETA-TWO)
            END WHERE           
         CASE("SCAD")
            WHERE (ABSBETA<RHO.OR.ABSBETA>=ETA*RHO)
               D2PEN = ZERO
            ELSEWHERE
               D2PEN = ONE/(ONE-ETA)
            END WHERE         
         END SELECT
      END IF
!
!     Second mixed derivative of penalty function
!
      IF (PRESENT(DPENDRHO)) THEN
         SELECT CASE(PENTYPE)
         CASE("ENET")
            DPENDRHO = (ETA-ONE)*ABSBETA+TWO-ETA
         CASE("LOG")
            DPENDRHO = ONE/(ETAC+ABSBETA)
            IF (CONTLOG) THEN
               DPENDRHO = DPENDRHO*(ONE-HALF*ETAC*DPENDRHO)
            END IF         
         CASE("MCP")
            WHERE (ABSBETA<RHO*ETA)
               DPENDRHO = ONE
            ELSEWHERE
               DPENDRHO = ZERO
            END WHERE
         CASE("POWER")
            WHERE (ABSBETA<EPS)
               DPENDRHO = ETA*EPS**(ETA-ONE)
            ELSEWHERE
               DPENDRHO = ETA*ABSBETA**(ETA-ONE)
            END WHERE         
         CASE("SCAD")
            WHERE (ABSBETA<RHO)
               DPENDRHO = ONE
            ELSEWHERE (ABSBETA<ETA*RHO)
               DPENDRHO = ETA/(ETA-ONE)
            ELSEWHERE
               DPENDRHO = ZERO
            END WHERE         
         END SELECT
      END IF
      END SUBROUTINE PENALTY_FUN
!
      FUNCTION LSQ_THRESHOLDING(A,B,RHO,ETA,PENTYPE) RESULT(XMIN)
!
!     This subroutine performs univariate soft thresholding for least squares:
!     argmin .5*a*x^2+b*x+PEN(abs(x),eta).
!
      IMPLICIT NONE
      CHARACTER(LEN=*), INTENT(IN) :: PENTYPE      
      LOGICAL :: DOBISECTION
      REAL(KIND=DBLE_PREC), PARAMETER :: EPS=1E-8
      REAL(KIND=DBLE_PREC), INTENT(IN) :: A,B,ETA,RHO
      REAL(KIND=DBLE_PREC) :: ABSB,BC,DL,DM,DR,ETAC,F1,F2,FXMIN,FXMIN2
      REAL(KIND=DBLE_PREC) :: ROOT,UB,XL,XM,XMIN,XMIN2,XR
!
!     Check tuning parameter
!
      IF (RHO<ZERO) THEN
         !PRINT*, "PENALTY TUNING CONSTANT SHOULD BE NONNEGATIVE"
         RETURN
      END IF
!
!     Transform to format 0.5*a*(x-b)^2
!
      IF (A<=ZERO) THEN
         !PRINT*, "QUADRATIC COEFFICIENT A MUST BE POSITIVE"
         RETURN
      END IF
      BC = -B/A
      ABSB = ABS(BC)
!
!     Thresholding
!
      IF (RHO<EPS) THEN
         XMIN = BC
         RETURN
      ELSE
         SELECT CASE(PENTYPE)
         CASE("ENET")
            IF (ETA<ONE.OR.ETA>TWO) THEN
               !PRINT*,"THE ENET PARAMETER ETA SHOULD BE IN [1,2]."
               RETURN
            ELSEIF (ABS(ETA-TWO)<EPS) THEN
               XMIN = A*BC/(A+RHO)
               RETURN
            END IF
            XMIN = A*BC-RHO*(TWO-ETA)
            IF (XMIN>ZERO) THEN
               XMIN = XMIN/(A+RHO*(ETA-1))
               RETURN
            END IF
            XMIN = A*BC+RHO*(TWO-ETA)
            IF (XMIN<ZERO) THEN
               XMIN = XMIN/(A+RHO*(ETA-1))
               RETURN
            END IF
            XMIN = ZERO
         CASE("LOG")
            IF (ETA<ZERO) THEN
               !PRINT *, "PARAMETER ETA FOR LOG PENALTY SHOULD BE POSITIVE"
               RETURN
            ELSEIF (ABS(ETA)<EPS) THEN
               ETAC = SQRT(RHO)
            ELSE
               ETAC = ETA
            END IF
            IF (RHO<=A*ABSB*ETAC) THEN
               XMIN = HALF*(ABSB-ETAC+ &
                  SQRT(MAX((ABSB+ETAC)*(ABSB+ETAC)-FOUR*RHO/A,ZERO)))
            ELSEIF (RHO<=A*ETAC*ETAC) THEN
               XMIN = ZERO
            ELSEIF (RHO>=A*(ETAC+ABSB)*(ETAC+ABSB)/FOUR) THEN
               XMIN = ZERO
            ELSE
               XMIN = HALF*(ABSB-ETAC+ &
                  SQRT(MAX((ABSB+ETAC)*(ABSB+ETAC)-FOUR*RHO/A,ZERO)))
               F1 = HALF*A*BC*BC+RHO*LOG(ETAC)
               F2 = HALF*A*(XMIN-ABSB)**2+RHO*LOG(ETAC+ABS(XMIN))
               IF (F1<F2) THEN
                  XMIN = ZERO
               END IF
            END IF
            XMIN = SIGN(XMIN,BC)
         CASE("MCP")
            IF (ETA<=ZERO) THEN
               !PRINT*,"THE MCP PARAMETER ETA SHOULD BE POSITIVE."
               RETURN
            END IF
            UB = MIN(RHO*ETA,ABSB)
            IF (A*ETA==ONE) THEN
               IF (RHO>=A*ABSB) THEN
                  XMIN = ZERO
               ELSE
                  XMIN = UB
               END IF
            ELSEIF (A*ETA>ONE) THEN
               ROOT = -(RHO-A*ABSB)/(A-ONE/ETA)
               IF (ROOT<=ZERO) THEN
                  XMIN = ZERO
               ELSEIF (ROOT>=UB) THEN
                  XMIN = UB
               ELSE
                  XMIN = ROOT
               END IF
            ELSE
               ROOT = -(RHO-A*ABSB)/(A-ONE/ETA)
               IF (TWO*ROOT>=UB) THEN
                  XMIN = ZERO
               ELSE
                  XMIN = UB
               END IF
            END IF
            IF (RHO*ETA<ABSB) THEN
               IF (HALF*(A*(XMIN-ABSB)**2+RHO*XMIN-HALF*XMIN*XMIN/ETA)&
                  >(HALF*RHO*RHO*ETA)) THEN
                  XMIN = ABSB
               END IF
            END IF
            XMIN = SIGN(XMIN,BC)
         CASE("POWER")
            IF (ETA<=ZERO.OR.ETA>TWO) THEN
               !PRINT*,"THE EXPONENT ETA SHOULD BE IN (0,2]."
               RETURN
            ELSEIF (ABS(ETA-TWO)<EPS) THEN
               XMIN = A*BC/(A+TWO*RHO)
               RETURN
            ELSEIF (ABS(ETA-ONE)<EPS) THEN
               IF (BC-RHO/A>ZERO) THEN
                  XMIN = BC-RHO/A
               ELSEIF (BC+RHO/A<ZERO) THEN
                  XMIN = BC+RHO/A
               ELSE
                  XMIN = ZERO
               END IF
               RETURN
            END IF
            DOBISECTION = .FALSE.
            IF (ETA>ONE) THEN
               XL = ZERO
               DL = -A*ABSB
               DOBISECTION = .TRUE.
            ELSEIF (A+RHO*ETA*(ETA-ONE)*ABSB**(ETA-TWO)<=ZERO) THEN
               XMIN = ZERO
            ELSE
               XL = (A/RHO/ETA/(ONE-ETA))**(ONE/(ETA-TWO))
               DL = A*(XL-ABSB)+RHO*ETA*XL**(ETA-ONE)
               IF (DL>=ZERO) THEN
                  XMIN = ZERO
               ELSE
                  DOBISECTION = .TRUE.
               END IF
            END IF
            IF (DOBISECTION) THEN
               XR = ABSB
               DR = RHO*ETA*XR**(ETA-ONE)
               DO
                  XM = HALF*(XL+XR)
                  DM = A*(XM-ABSB)+RHO*ETA*XM**(ETA-ONE)
                  IF (DM>EPS) THEN
                     XR = XM
                     DR = DM
                  ELSEIF (DM<-EPS) THEN
                     XL = XM
                     DL = DM
                  ELSE
                     XMIN = XM
                     EXIT
                  END IF
                  IF (ABS(XL-XR)<EPS) THEN
                     XMIN = XM
                     EXIT
                  END IF
               END DO
               IF ((ETA<ONE) .AND. &
                  (HALF*A*(XMIN-ABSB)**2+RHO*ABS(XMIN)**ETA>HALF*A*ABSB**2)) THEN
                  XMIN = ZERO
               END IF
            END IF
            XMIN = SIGN(XMIN,BC)
         CASE("SCAD")
            IF (ETA<=TWO) THEN
               !PRINT*,"THE SCAD PARAMETER ETA SHOULD BE GREATER THAN 2."
               RETURN
            END IF 
!            IF (RHO<=A*ABSB) THEN
!               IF (A*(RHO-ABSB)+RHO>=ZERO) THEN
!                  XMIN = ABSB - RHO/A
!               ELSEIF (RHO*ETA-ABSB>=ZERO) THEN
!                  XMIN = (A*ABSB*(ETA-ONE)-RHO*ETA)/(A*(ETA-ONE)-ONE)
!               ELSE
!                  XMIN = ABSB
!               END IF
!            ELSEIF (ABSB<=RHO*ETA.OR.A*(ETA-ONE)>=ONE) THEN
!               XMIN = ZERO
!            ELSE
!               IF (A*BC*BC<RHO*RHO*(ETA+ONE)) THEN
!                  XMIN = ZERO
!               ELSE
!                  XMIN = ABSB
!               END IF
!            END IF
!            XMIN = SIGN(XMIN,BC)
!
!     Compute minimum on [0,rho]
!
            UB = MIN(ABSB,RHO)
            ROOT = ABSB-RHO/A
            IF (ROOT<=ZERO) THEN
               XMIN = ZERO
            ELSE
               XMIN = MIN(ROOT,UB)
            END IF
            IF (ABSB<=RHO) THEN
               XMIN = SIGN(XMIN,BC)
               RETURN
            END IF
!
!     Compute minimum on [rho,eta*rho]
!            
            UB = MIN(ETA*RHO,ABSB)
            ROOT = (A*ABSB*(ETA-ONE)-ETA*RHO)/(A*(ETA-ONE)-ONE)
            IF (A*(ETA-ONE)<ONE) THEN
               IF (TWO*ROOT>=RHO+UB) THEN
                  XMIN2 = RHO
               ELSE
                  XMIN2 = UB
               END IF
            ELSE IF (A*(ETA-ONE)>ONE) THEN
               IF (ROOT<=RHO) THEN
                  XMIN2 = RHO
               ELSE IF (ROOT>UB) THEN
                  XMIN2 = UB
               ELSE
                  XMIN2 = ROOT
               END IF
            ELSE
               IF (ETA*RHO>=ABSB) THEN
                  XMIN2 = RHO
               ELSE
                  XMIN2 = UB
               END IF
            END IF
            F1 = HALF*(XMIN-ABSB)**2 + RHO*XMIN
            F2 = HALF*(XMIN2-ABSB)**2 + RHO**2 + ETA*RHO*(XMIN2-RHO)/(ETA-ONE) &
               - HALF*(XMIN2**2-RHO**2)/(ETA-ONE)
            IF (F2<F1) THEN
               XMIN = XMIN2
               F1 = F2
            END IF
            IF (ABSB<=ETA*RHO) THEN
               XMIN = SIGN(XMIN,BC)
               RETURN
            END IF
!
!     Compute minimum on (eta*rho,infty)
!                        
            XMIN2 = ABSB
            F2 = HALF*RHO**2*(ETA+ONE)
            IF (F2<F1) THEN
               XMIN = XMIN2
            END IF
            XMIN = SIGN(XMIN,BC)
         END SELECT
      END IF
      END FUNCTION LSQ_THRESHOLDING
!
      FUNCTION GLM_THRESHOLDING(X,C,Y,WT,RHO,ETA,PENTYPE,MODEL) RESULT(BETA)
!
!     This subroutine performs univariate thresholding for GLM with linear
!     part X*BETA+C: argmin loss(beta)+PEN(abs(beta),rho,eta).
!
      IMPLICIT NONE
      CHARACTER(LEN=*), INTENT(IN) :: MODEL,PENTYPE
      REAL(KIND=DBLE_PREC), INTENT(IN) :: ETA,RHO
      REAL(KIND=DBLE_PREC), DIMENSION(:), INTENT(IN) :: C,WT,X,Y
!
!     Local variables
!
      INTEGER, PARAMETER :: MAXITERS=50
      INTEGER :: ITERATION
      REAL(KIND=DBLE_PREC), PARAMETER :: CONVCRIT=1E-4
      REAL(KIND=DBLE_PREC) :: BETA,D1,D2,LOSS,LOSS_OLD
!      
!     Check tuning parameter
!
      IF (RHO<ZERO) THEN
         !PRINT*, "PENALTY TUNING CONSTANT SHOULD BE NONNEGATIVE"
         RETURN
      END IF
!
!     Iteratively reweighted penalized least squares
!
      BETA = ZERO
      LOSS = 1E16
      DO ITERATION=1,MAXITERS
         LOSS_OLD = LOSS
         CALL SIMPLE_GLM_LOSS(BETA,X,C,Y,WT,MODEL,LOSS,D1,D2) 
         BETA = LSQ_THRESHOLDING(D2,D1-BETA*D2,RHO,ETA,PENTYPE)
         IF (ABS(LOSS_OLD-LOSS)<CONVCRIT*(ABS(LOSS_OLD)+ONE)) THEN
            EXIT
         END IF
      END DO
      END FUNCTION GLM_THRESHOLDING
!
      SUBROUTINE SIMPLE_GLM_LOSS(BETA,X,C,Y,WT,MODEL,LOSS,D1,D2)
!
!     This subroutine computes the loss, derivative, and second derivative
!     of a GLM model with linear part: X*BETA+C
!
      IMPLICIT NONE
      CHARACTER(LEN=*), INTENT(IN) :: MODEL
      REAL(KIND=DBLE_PREC), INTENT(IN) :: BETA
      REAL(KIND=DBLE_PREC), INTENT(OUT) :: LOSS
      REAL(KIND=DBLE_PREC), INTENT(OUT), OPTIONAL :: D1,D2
      REAL(KIND=DBLE_PREC), DIMENSION(:), INTENT(IN) :: C,WT,X,Y
!
!     Local variables
!      
      INTEGER :: I,N
      REAL(KIND=DBLE_PREC), PARAMETER :: BIG=TWO*TEN
      REAL(KIND=DBLE_PREC), DIMENSION(SIZE(X)) :: EXPINNER,INNER,PROB
!
!     Check argument
!
      N = SIZE(X)      
      IF (SIZE(C)/=N) THEN
         !PRINT*, " SIZE OF C INCOMPATIBLE WITH X"
         RETURN
      END IF
      IF (SIZE(Y)/=N) THEN
         !PRINT*, " SIZE OF Y INCOMPATIBLE WITH X"
         RETURN
      END IF
      IF (SIZE(WT)/=N) THEN
         !PRINT*, " SIZE OF WT INCOMPATIBLE WITH X"
         RETURN
      END IF      
!
!     Compute the linear parts and its exponential
!
      INNER = BETA*X+C
      EXPINNER = EXP(INNER)
!
!     Compute loss function
!
      SELECT CASE(MODEL)
      CASE("LOGISTIC")
         WHERE (INNER>=BIG)
            PROB = INNER
         ELSEWHERE (INNER<=-BIG)
            PROB = ZERO
         ELSEWHERE
            PROB = LOG(ONE+EXPINNER)
         END WHERE
         LOSS = - SUM(WT*(Y*INNER-PROB))
      CASE("LOGLINEAR")
         LOSS = - SUM(WT*(Y*INNER-EXPINNER))
      END SELECT
!
!     Compute first derivative
!
      IF (PRESENT(D1)) THEN
         SELECT CASE(MODEL)
         CASE("LOGISTIC")
            WHERE (INNER>=BIG)
               PROB = ONE
            ELSEWHERE (INNER<=-BIG)
               PROB = ZERO
            ELSEWHERE
               PROB = EXPINNER/(ONE+EXPINNER)
            END WHERE
            D1 = - SUM(WT*(Y-PROB)*X)
         CASE("LOGLINEAR")
            D1 = - SUM(WT*(Y-EXPINNER)*X)
         END SELECT
      END IF
!
!     Compute second derivative
!
      IF (PRESENT(D2)) THEN
         SELECT CASE(MODEL)
         CASE("LOGISTIC")
            D2 = SUM(WT*PROB*(ONE-PROB)*X*X)
         CASE("LOGLINEAR")
            D2 = SUM(WT*EXPINNER*X*X)
         END SELECT
      END IF
      END SUBROUTINE SIMPLE_GLM_LOSS
!
      FUNCTION MAX_RHO(A,B,PENTYPE,PENPARAM) RESULT(MAXRHO)
!
!     This subroutine finds the maximum penalty constant rho such that
!     argmin 0.5*A*x^2+B*x+penalty(x,rho) is nonzero. Current options for
!     PENTYPE are "ENET","LOG","MCP","POWER","SCAD". PENPARAM contains the
!     optional parameter for the penalty function.
!
      IMPLICIT NONE
      CHARACTER(LEN=*), INTENT(IN) :: PENTYPE
      REAL(KIND=DBLE_PREC), INTENT(IN) :: A,B
      REAL(KIND=DBLE_PREC), DIMENSION(:), INTENT(IN) :: PENPARAM
!
!     Local variables
!
      REAL(KIND=DBLE_PREC), PARAMETER :: EPS=1E-8
      REAL(KIND=DBLE_PREC) :: BC,ETA,L,M,MAXRHO,R,ROOTL,ROOTM,ROOTR
!
!     Locate the max rho
!
      ETA = PENPARAM(1)
      SELECT CASE(PENTYPE)
      CASE("ENET")
         IF (PENPARAM(1)==TWO) THEN
            MAXRHO = TEN*A
         ELSE
            MAXRHO = ABS(B)/(TWO-PENPARAM(1))
         END IF
      CASE("LOG")
         IF (ABS(ETA)<EPS) THEN
            MAXRHO = B*B
            RETURN
         ELSE
            L = ABS(B*ETA)
            R = MAX(ETA*ABS(B),A*ETA*ETA,A*(ETA+ABS(B)/A)**2/FOUR)
         END IF
         ROOTL = LSQ_THRESHOLDING(A,B,L,ETA,"LOG")
         ROOTR = LSQ_THRESHOLDING(A,B,R,ETA,"LOG")
         DO
            M = HALF*(L+R)
            ROOTM = LSQ_THRESHOLDING(A,B,M,ETA,"LOG")
            IF (ROOTM==ZERO) THEN
               R = M
               ROOTR = ROOTM
            ELSE
               L = M
               ROOTL = ROOTM
            END IF
            IF (ABS(R-L)<EPS) THEN
               MAXRHO = M
               EXIT
            END IF
         END DO
         RETURN                  
      CASE("MCP")
         L = ABS(B)
         IF (A*PENPARAM(1)>=ZERO) THEN
            R = ABS(B)
         ELSE
            R = MAX(ABS(B)/PENPARAM(1),A*ABS(B))
         END IF
         ROOTL = LSQ_THRESHOLDING(A,B,L,PENPARAM(1),"MCP")
         ROOTR = LSQ_THRESHOLDING(A,B,R,PENPARAM(1),"MCP")
         DO
            M = HALF*(L+R)
            ROOTM = LSQ_THRESHOLDING(A,B,M,PENPARAM(1),"MCP")
            IF (ROOTM==ZERO) THEN
               R = M
               ROOTR = ROOTM
            ELSE
               L = M
               ROOTL = ROOTM
            END IF
            IF (ABS(R-L)<EPS) THEN
               MAXRHO = M
               EXIT
            END IF
         END DO
      CASE("POWER")
         IF (ABS(ETA-ONE)<EPS) THEN
            MAXRHO = ABS(B)
            RETURN
         ELSEIF (ETA<ONE) THEN
            L = ZERO
            ROOTL = ABS(LSQ_THRESHOLDING(A,B,L,ETA,"POWER"))
            R = A/ETA/(ONE-ETA)/ABS(B/A)**(ETA-TWO)
            ROOTR = ZERO
            DO
               M = HALF*(L+R)
               ROOTM = ABS(LSQ_THRESHOLDING(A,B,M,ETA,"POWER"))
               IF (ROOTM==ZERO) THEN
                  R = M
                  ROOTR = ROOTM
               ELSE
                  L = M
                  ROOTL = ROOTM
               END IF
               IF (ABS(R-L)<EPS) THEN
                  MAXRHO = M
                  EXIT
               END IF
            END DO
            RETURN
         ELSEIF (ETA>ONE) THEN
            R = ONE
            ROOTR = LSQ_THRESHOLDING(A,B,R,ETA,"POWER")
            DO WHILE(ABS(ROOTR)>ABS(B)/A/TEN)
               R = TWO*R
               ROOTR = LSQ_THRESHOLDING(A,B,R,ETA,"POWER")
            END DO
            MAXRHO = R
            RETURN
         END IF
      CASE("SCAD")
         BC = -B/A
         L = A/(A+ONE)*ABS(BC)
         R = A*ABS(BC)
         ROOTL = LSQ_THRESHOLDING(A,B,L,PENPARAM(1),"SCAD")
         ROOTR = LSQ_THRESHOLDING(A,B,R,PENPARAM(1),"SCAD")
         DO
            M = HALF*(L+R)
            ROOTM = LSQ_THRESHOLDING(A,B,M,PENPARAM(1),"SCAD")
            IF (ROOTM==ZERO) THEN
               R = M
               ROOTR = ROOTM
            ELSE
               L = M
               ROOTL = ROOTM
            END IF
            IF (ABS(R-L)<EPS) THEN
               MAXRHO = M
               EXIT
            END IF
         END DO
         RETURN
      END SELECT
      END FUNCTION MAX_RHO
!
      FUNCTION GLM_MAXRHO(X,C,Y,WT,PENTYPE,PENPARAM,MODEL) RESULT(MAXRHO)
!
!     This subroutine finds the maximum penalty constant rho such that
!     argmin loss(x)+penalty(x,rho) is nonzero. Current options for
!     PENTYPE are "ENET","LOG","MCP","POWER","SCAD". PENPARAM contains the
!     optional parameter for the penalty function.
!
      IMPLICIT NONE
      CHARACTER(LEN=*), INTENT(IN) :: MODEL,PENTYPE
      REAL(KIND=DBLE_PREC), DIMENSION(:), INTENT(IN) :: C,PENPARAM,WT,X,Y
!
!     Local variables
!
      REAL(KIND=DBLE_PREC) :: D1,D2,LOSS,MAXRHO
!
!     Obtain derivatives of beta
!      
      CALL SIMPLE_GLM_LOSS(ZERO,X,C,Y,WT,MODEL,LOSS,D1,D2)           
      MAXRHO = MAX_RHO(D2,D1,PENTYPE,PENPARAM)
      END FUNCTION GLM_MAXRHO
!
      SUBROUTINE PENALIZED_L2_REGRESSION(ESTIMATE,X,Y,WT,LAMBDA, &
         SUM_X_SQUARES,PENIDX,MAXITERS,PENTYPE,PENPARAM)
!
!     This subroutine carries out penalized L2 regression with design
!     matrix X, dependent variable Y, weights W and penalty constant 
!     LAMBDA. Note that the rows of X correspond to cases and the columns
!     to predictors.  The SUM_X_SQUARES should have entries SUM(W*X(:,I)**2)
!
      IMPLICIT NONE
      CHARACTER(LEN=*), INTENT(IN) :: PENTYPE
      INTEGER :: MAXITERS
      LOGICAL, INTENT(IN), DIMENSION(:) :: PENIDX
      REAL(KIND=DBLE_PREC), INTENT(IN) :: LAMBDA
      REAL(KIND=DBLE_PREC), INTENT(IN), DIMENSION(:) :: PENPARAM,SUM_X_SQUARES,WT,Y
      REAL(KIND=DBLE_PREC), INTENT(INOUT), DIMENSION(:) :: ESTIMATE
      REAL(KIND=DBLE_PREC), INTENT(IN), DIMENSION(:,:) :: X
!
!     Local variables.
!
      INTEGER :: I,ITERATION,M,N
      LOGICAL, DIMENSION(SIZE(ESTIMATE)) :: NZIDX
      REAL(KIND=DBLE_PREC), PARAMETER :: CRITERION=1E-8,EPS=1E-8
      REAL(KIND=DBLE_PREC) :: A,B,NEW_OBJECTIVE,OLDROOT,OBJECTIVE,ROOTDIFF
      REAL(KIND=DBLE_PREC), DIMENSION(SIZE(ESTIMATE)) :: PENALTY
      REAL(KIND=DBLE_PREC), DIMENSION(SIZE(Y)) :: R
!
!     Check that the number of cases is well defined.
!
      IF (SIZE(Y)/=SIZE(X,1)) THEN
         !PRINT*," THE NUMBER OF CASES IS NOT WELL DEFINED."
         RETURN
      END IF
!
!     Check that the number of predictors is well defined.
!
      IF (SIZE(ESTIMATE)/=SIZE(X,2)) THEN
         !PRINT*, " THE NUMBER OF PREDICTORS IS NOT WELL DEFINED."
         RETURN
      END IF
!
!     Check the index for penalized predictors
!
      IF (SIZE(PENIDX)/=SIZE(X,2)) THEN
         !PRINT*, " THE PENALTY INDEX ARRAY IS NOT WELL DEFINED"
         RETURN
      END IF
!
!     Initialize the number of cases M and the number of regression
!     coefficients N.
!
      M = SIZE(Y)
      N = SIZE(ESTIMATE)
!
!     Initialize the residual vector R, the PENALTY, the loss L2, and the
!     objective function.
!
      IF (ANY(ABS(ESTIMATE)>EPS)) THEN
         R = Y-MATMUL(X,ESTIMATE)
      ELSE
         R = Y
      END IF
      CALL PENALTY_FUN(ESTIMATE,LAMBDA,PENPARAM(1),PENTYPE,PENALTY)
      OBJECTIVE = HALF*SUM(WT*R*R)+SUM(PENALTY,PENIDX)
      !PRINT*, "OBJECTIVE = "
      !PRINT*, OBJECTIVE
!
!     Initialize maximum number of iterations
!
      IF (MAXITERS<=0) THEN
         MAXITERS = 1000
      END IF
!
!     Enter the main iteration loop.
!
      DO ITERATION = 1,MAXITERS
         DO I = 1,N
            OLDROOT = ESTIMATE(I)           
            A = SUM_X_SQUARES(I)
            B = - SUM(WT*R*X(:,I)) - A*OLDROOT
            IF (PENIDX(I)) THEN
               ESTIMATE(I) = LSQ_THRESHOLDING(A,B,LAMBDA,PENPARAM(1),PENTYPE)
            ELSE
               ESTIMATE(I) = -B/A
            END IF
            ROOTDIFF = ESTIMATE(I)-OLDROOT
            IF (ABS(ROOTDIFF)>EPS) THEN
               R = R - ROOTDIFF*X(:,I)
            END IF
         END DO
!
!     Output the iteration number and value of the objective function.
!
         CALL PENALTY_FUN(ESTIMATE,LAMBDA,PENPARAM(1),PENTYPE,PENALTY)
         NEW_OBJECTIVE = HALF*SUM(WT*R*R)+SUM(PENALTY,PENIDX)
         IF (ITERATION==1.OR.MOD(ITERATION,1)==0) THEN
            !PRINT*," ITERATION = ",ITERATION," FUN = ",NEW_OBJECTIVE
         END IF
!
!     Check for a descent failure or convergence.  If neither occurs,
!     record the new value of the objective function.
!
         IF (NEW_OBJECTIVE>OBJECTIVE+EPS) THEN
            !PRINT*," *** ERROR *** OBJECTIVE FUNCTION INCREASE AT ITERATION",ITERATION
            RETURN
         END IF
         IF ((OBJECTIVE-NEW_OBJECTIVE)<CRITERION*(ABS(OBJECTIVE)+ONE)) THEN
            RETURN
         ELSE
            OBJECTIVE = NEW_OBJECTIVE
         END IF
      END DO
      END SUBROUTINE PENALIZED_L2_REGRESSION
!
      SUBROUTINE PENALIZED_GLM_REGRESSION(ESTIMATE,X,Y,WT,LAMBDA, &
         PENIDX,MAXITERS,PENTYPE,PENPARAM,MODEL)
!
!     This subroutine carries out penalized GLM regression with design
!     matrix X, dependent variable Y, weights WT and penalty constant 
!     LAMBDA, using iteratively reweighted least squares (IRLS).
!
      IMPLICIT NONE
      CHARACTER(LEN=*), INTENT(IN) :: PENTYPE,MODEL
      REAL(KIND=DBLE_PREC), INTENT(IN) :: LAMBDA
      LOGICAL, DIMENSION(:), INTENT(IN) :: PENIDX
      REAL(KIND=DBLE_PREC), DIMENSION(:), INTENT(IN) :: PENPARAM,WT,Y
      REAL(KIND=DBLE_PREC), DIMENSION(:), INTENT(INOUT) :: ESTIMATE
      REAL(KIND=DBLE_PREC), DIMENSION(:,:), INTENT(IN) :: X
!
!     Local variables
!      
      INTEGER :: ITERATION,MAXITERS
      REAL(KIND=DBLE_PREC), PARAMETER :: BIG=2E1,CRITERION=1E-4,EPS=1E-8
      REAL(KIND=DBLE_PREC) :: LOSS,NEW_OBJECTIVE,OBJECTIVE
      REAL(KIND=DBLE_PREC), DIMENSION(SIZE(Y)) :: EXPINNER,INNER,IRLSWT,IRLSY,MU
      REAL(KIND=DBLE_PREC), DIMENSION(SIZE(ESTIMATE)) :: PENALTY
      REAL(KIND=DBLE_PREC), DIMENSION(SIZE(X,1),SIZE(X,2)) :: X_SQUARE
!
!     Check that the number of cases is well defined.
!
      IF (SIZE(Y)/=SIZE(X,1)) THEN
         !PRINT*," THE NUMBER OF CASES IS NOT WELL DEFINED."
         RETURN
      END IF
!
!     Check that the number of predictors is well defined.
!
      IF (SIZE(ESTIMATE)/=SIZE(X,2)) THEN
         !PRINT*, " THE NUMBER OF PREDICTORS IS NOT WELL DEFINED."
         RETURN
      END IF
!
!     Check the index for penalized predictors
!
      IF (SIZE(PENIDX)/=SIZE(X,2)) THEN
         !PRINT*, " THE PENALTY INDEX ARRAY IS NOT WELL DEFINED"
         RETURN
      END IF
!
!     Precompute elementwise square of design matrix
!
      X_SQUARE = X**2
!
!     Compute the loss, IRLS weights, and penalty values
!      
      IF (ANY(ABS(ESTIMATE)>EPS)) THEN
         INNER = MATMUL(X,ESTIMATE)
      ELSE
         INNER = ZERO
      END IF
      EXPINNER = EXP(INNER)
      SELECT CASE(MODEL)
      CASE("LOGISTIC")
         WHERE (INNER>=BIG)
            MU = ONE-EPS
         ELSEWHERE (INNER<=-BIG)
            MU = EPS
         ELSEWHERE
            MU = EXPINNER/(ONE+EXPINNER)
         END WHERE
         LOSS = - SUM(WT*LOG(Y*MU+(ONE-Y)*(ONE-MU)))
         IRLSWT = WT*MU*(ONE-MU)
         IRLSY = INNER+(Y-MU)/IRLSWT
      CASE("LOGLINEAR")
         LOSS = - SUM(WT*(Y*INNER-EXPINNER))
         WHERE (EXPINNER<EPS)
            IRLSWT = EPS
         ELSEWHERE
            IRLSWT = EXPINNER
         END WHERE
         IRLSWT = WT*IRLSWT
         IRLSY = INNER+(Y-EXPINNER)/IRLSWT
      END SELECT
      CALL PENALTY_FUN(ESTIMATE,LAMBDA,PENPARAM(1),PENTYPE,PENALTY)
      OBJECTIVE = LOSS+SUM(PENALTY,PENIDX)
      !PRINT*, "OBJECTIVE = "
      !PRINT*, OBJECTIVE
!
!     Initialize maximum number of iterations
!
      IF (MAXITERS<=0) THEN
         MAXITERS = 1000
      END IF
!
!     Enter the main iteration loop.
!
      DO ITERATION = 1,MAXITERS
!
!     Solve penalized weighted least squares
!
         CALL PENALIZED_L2_REGRESSION(ESTIMATE,X,IRLSY,IRLSWT,LAMBDA, &
            MATMUL(IRLSWT,X_SQUARE),PENIDX,5,PENTYPE,PENPARAM)
!
!     Update IRLS weights and loss function
!
         INNER = MATMUL(X,ESTIMATE)
         EXPINNER = EXP(INNER)
         SELECT CASE(MODEL)
         CASE("LOGISTIC")
            WHERE (INNER>=BIG)
               MU = ONE-EPS
            ELSEWHERE (INNER<=-BIG)
               MU = EPS
            ELSEWHERE
               MU = EXPINNER/(ONE+EXPINNER)
            END WHERE
            LOSS = - SUM(WT*LOG(Y*MU+(ONE-Y)*(ONE-MU)))
            IRLSWT = WT*MU*(ONE-MU)
            IRLSY = INNER+(Y-MU)/IRLSWT
         CASE("LOGLINEAR")
            LOSS = - SUM(WT*(Y*INNER-EXPINNER))
            WHERE (EXPINNER<EPS)
               IRLSWT = EPS
            ELSEWHERE
               IRLSWT = EXPINNER
            END WHERE
            IRLSWT = WT*IRLSWT
            IRLSY = INNER+(Y-EXPINNER)/IRLSWT
         END SELECT
         CALL PENALTY_FUN(ESTIMATE,LAMBDA,PENPARAM(1),PENTYPE,PENALTY)
         NEW_OBJECTIVE = LOSS+SUM(PENALTY,PENIDX)
         IF (ITERATION==1.OR.MOD(ITERATION,1)==0) THEN
            !PRINT*," ITERATION = ",ITERATION," FUN = ",NEW_OBJECTIVE
         END IF
!
!     Record the new value of the objective function.
!
         IF (ABS(OBJECTIVE-NEW_OBJECTIVE)<CRITERION*(ABS(OBJECTIVE)+ONE)) THEN
            RETURN
         ELSE
            OBJECTIVE = NEW_OBJECTIVE
         END IF
      END DO
      END SUBROUTINE PENALIZED_GLM_REGRESSION
!
      END MODULE SPARSEREG
!