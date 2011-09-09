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
      SUBROUTINE READ_DATA(INPUT_FILE,X,Y,N,P)
!
!     THIS SUBROUTINE READS IN THE DATA AND INITIALIZES CONSTANTS AND ARRAYS.
!
      IMPLICIT NONE
      CHARACTER(LEN=100) :: INPUT_FILE
      INTEGER :: I,INPUT_UNIT=1,N,P
      REAL(KIND=DBLE_PREC), DIMENSION(:) :: Y
      REAL(KIND=DBLE_PREC), DIMENSION(:,:) :: X
!
!     Open the input file.
!
      OPEN(UNIT=INPUT_UNIT,FILE=INPUT_FILE)
!
!     Read the exponents
!
      DO I = 1,N
	      READ(INPUT_UNIT,*) Y(I),X(I,2:P)
	      X(I,1) = ONE
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
      LOGICAL :: CONTLOG
      REAL(KIND=DBLE_PREC), PARAMETER :: EPS=1E-8
      REAL(KIND=DBLE_PREC) :: ETA,RHO
      REAL(KIND=DBLE_PREC), DIMENSION(:) :: BETA,PEN
      REAL(KIND=DBLE_PREC), DIMENSION(SIZE(BETA)) :: ABSBETA
      REAL(KIND=DBLE_PREC), OPTIONAL, DIMENSION(:) :: D1PEN,D2PEN,DPENDRHO
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
         IF (ETA==ZERO) THEN
            ETA = SQRT(RHO)
            CONTLOG = .TRUE.
         ELSEIF (ETA<ZERO) THEN
            !PRINT*,"THE LOG PENALTY PARAMETER ETA SHOULD BE NONNEGATIVE."
            RETURN
         END IF
         PEN = RHO*LOG(ETA+ABSBETA)      
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
            D1PEN = RHO/(ETA+ABSBETA)
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
            D2PEN = -D1PEN/(ETA+ABSBETA)
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
            WHERE (ABSBETA<RHO.OR.ABSBETA>ETA*RHO)
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
            DPENDRHO = ONE/(ETA+ABSBETA)
            IF (CONTLOG) THEN
               DPENDRHO = DPENDRHO*(ONE-HALF*ETA*DPENDRHO)
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
!        argmin .5*a*x^2+b*x+PEN(abs(x),eta).
!
      IMPLICIT NONE
      CHARACTER(LEN=*), INTENT(IN) :: PENTYPE      
      LOGICAL :: DOBISECTION
      REAL(KIND=DBLE_PREC), PARAMETER :: EPS=1E-8
      REAL(KIND=DBLE_PREC) :: A,ABSB,B,DL,DM,DR,ETA,F1,F2,FXMIN,FXMIN2
      REAL(KIND=DBLE_PREC) :: RHO,XL,XM,XMIN,XMIN2,XR
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
      B = -B/A
      ABSB = ABS(B)
!
!     Thresholding
!
      IF (RHO<EPS) THEN
         XMIN = B
         RETURN
      ELSE
         SELECT CASE(PENTYPE)
         CASE("ENET")
            IF (ETA<ONE.OR.ETA>TWO) THEN
               !PRINT*,"THE ENET PARAMETER ETA SHOULD BE IN [1,2]."
               RETURN
            ELSEIF (ABS(ETA-TWO)<EPS) THEN
               XMIN = A*B/(A+RHO)
               RETURN
            END IF
            XMIN = A*B-RHO*(TWO-ETA)
            IF (XMIN>ZERO) THEN
               XMIN = XMIN/(A+RHO*(ETA-1))
               RETURN
            END IF
            XMIN = A*B+RHO*(TWO-ETA)
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
               ETA = SQRT(RHO)
            END IF
            IF (RHO<=A*ABSB*ETA) THEN
               XMIN = HALF*(ABSB-ETA+ &
                  SQRT((ABSB+ETA)*(ABSB+ETA)-FOUR*RHO/A))
            ELSEIF (RHO<=A*ETA*ETA) THEN
               XMIN = ZERO
            ELSEIF (RHO>=A*(ETA+ABSB)*(ETA+ABSB)/FOUR) THEN
               XMIN = ZERO
            ELSE
               XMIN = HALF*(ABSB-ETA+ &
                  SQRT((ABSB+ETA)*(ABSB+ETA)-FOUR*RHO/A))
               F1 = HALF*A*B*B+RHO*LOG(ETA)
               F2 = HALF*A*(XMIN-ABSB)**2+RHO*LOG(ETA+ABS(XMIN))
               IF (F1<F2) THEN
                  XMIN = ZERO
               END IF
            END IF
            XMIN = SIGN(XMIN,B)
         CASE("MCP")
            IF (ETA<=ZERO) THEN
               !PRINT*,"THE MCP PARAMETER ETA SHOULD BE POSITIVE."
               RETURN
            END IF
            IF (RHO<=A*ABSB) THEN
               IF (ABSB<=RHO*ETA) THEN
                  XMIN = ETA*(A*ABSB-RHO)/(A*ETA-ONE)
               ELSE
                  XMIN = ABSB
               END IF
            ELSEIF (A*ETA>=ONE) THEN
               XMIN = ZERO
            ELSEIF (RHO*ETA>=ABSB) THEN
               XMIN = ZERO
            ELSE
               IF (A*B*B<RHO*RHO*ETA) THEN
                  XMIN = ZERO
               ELSE
                  XMIN = ABSB
               END IF
            END IF
            XMIN = SIGN(XMIN,B)
         CASE("POWER")
            IF (ETA<=ZERO.OR.ETA>TWO) THEN
               !PRINT*,"THE EXPONENT ETA SHOULD BE IN (0,2]."
               RETURN
            ELSEIF (ABS(ETA-TWO)<EPS) THEN
               XMIN = A*B/(A+TWO*RHO)
               RETURN
            ELSEIF (ABS(ETA-ONE)<EPS) THEN
               IF (B-RHO/A>ZERO) THEN
                  XMIN = B-RHO/A
               ELSEIF (B+RHO/A<ZERO) THEN
                  XMIN = B+RHO/A
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
            XMIN = SIGN(XMIN,B)
         CASE("SCAD")
            IF (ETA<=TWO) THEN
               !PRINT*,"THE SCAD PARAMETER ETA SHOULD BE GREATER THAN 2."
               RETURN
            END IF 
            IF (RHO<=A*ABSB) THEN
               IF (A*(RHO-ABSB)+RHO>=ZERO) THEN
                  XMIN = ABSB - RHO/A
               ELSEIF (RHO*ETA-ABSB>=ZERO) THEN
                  XMIN = (A*ABSB*(ETA-ONE)-RHO*ETA)/(A*(ETA-ONE)-ONE)
               ELSE
                  XMIN = ABSB
               END IF
            ELSEIF (ABSB<=RHO*ETA.OR.A*(ETA-ONE)>=ONE) THEN
               XMIN = ZERO
            ELSE
               IF (A*B*B<RHO*RHO*(ETA+ONE)) THEN
                  XMIN = ZERO
               ELSE
                  XMIN = ABSB
               END IF
            END IF
            XMIN = SIGN(XMIN,B)
         END SELECT
      END IF
      END FUNCTION LSQ_THRESHOLDING
!
      FUNCTION GLM_THRESHOLDING(X,C,Y,WT,RHO,ETA,PENTYPE,MODEL) RESULT(XMIN)
!
!     This subroutine performs univariate thresholding for GLM with linear
!     part X*BETA+C: argmin loss(beta)+PEN(abs(x),rho,eta).
!
      IMPLICIT NONE
      CHARACTER(LEN=*), INTENT(IN) :: MODEL,PENTYPE
      INTEGER, PARAMETER :: BRACKETS=10
      INTEGER :: I,IDX
      LOGICAL :: DOBISECTION,ISNEGROOT
      LOGICAL, DIMENSION(BRACKETS) :: NEGIDX
      REAL(KIND=DBLE_PREC), PARAMETER :: EPS=1E-8
      REAL(KIND=DBLE_PREC) :: BETA,DELTAX,ETA,LOSSD2,PEN,PEND1,RHO,XMIN
      REAL(KIND=DBLE_PREC), DIMENSION(:), INTENT(IN) :: C,WT,X,Y
      REAL(KIND=DBLE_PREC), DIMENSION(3) :: XBRACKET,D1BRACKET
      REAL(KIND=DBLE_PREC), DIMENSION(BRACKETS) :: BETAVEC,LOSSVEC,LOSSD1VEC
      REAL(KIND=DBLE_PREC), DIMENSION(BRACKETS) :: PENVEC,PEND1VEC
!
!     Check tuning parameter
!
      IF (RHO<ZERO) THEN
         !PRINT*, "PENALTY TUNING CONSTANT SHOULD BE NONNEGATIVE"
         RETURN
      END IF
!
!     LOCATE THE UNPENALIZED ESTIMATE BY NEWTON-RALPHSON
!
      BETA = ZERO
      DO I=1,BRACKETS
         BETAVEC(I) = BETA
         CALL SIMPLE_GLM_LOSS(BETA,X,C,Y,WT,MODEL,LOSSVEC(I),LOSSD1VEC(I),LOSSD2)
         BETA = BETA - LOSSD1VEC(I)/LOSSD2
      END DO
      IF (ABS(RHO)<EPS) THEN
         XMIN = BETA
         RETURN
      END IF
!
!     Flip to positive axis if necessary
!
      IF (BETA<ZERO) THEN
         ISNEGROOT = .TRUE.
         BETA = -BETA
         BETAVEC = -BETAVEC
         LOSSD1VEC = -LOSSD1VEC
      ELSE
         ISNEGROOT = .FALSE.
      END IF
!
!     Search for negative derivative and use as bracket for bisection
!
      CALL PENALTY_FUN(BETAVEC,RHO,ETA,PENTYPE,PENVEC,PEND1VEC)
      NEGIDX = (BETAVEC>=ZERO).AND.(BETAVEC<=BETA).AND.(LOSSD1VEC+PEND1VEC<-EPS)
      IF (ANY(NEGIDX)) THEN
         DOBISECTION = .TRUE.
         IDX = MAXLOC(BETAVEC,1,NEGIDX)
         XBRACKET(1) = BETAVEC(IDX)
         D1BRACKET(1) = LOSSD1VEC(IDX) + PEND1VEC(IDX)
         IDX = MIN(MINLOC(BETAVEC,1,(BETAVEC>=XBRACKET(1)).AND.(LOSSD1VEC+PEND1VEC>=-EPS)),BRACKETS)
         XBRACKET(3) = BETAVEC(IDX)
         D1BRACKET(3) = LOSSD1VEC(IDX) + PEND1VEC(IDX)
      ELSE
!
!     Grid search for negative derivative
!
         DELTAX = BETA/(BRACKETS-1)
         DO I=2,BRACKETS-1
            BETAVEC(I) = DELTAX*(I-1)
            IF (ISNEGROOT) THEN
               CALL SIMPLE_GLM_LOSS(BETAVEC(I),-X,C,Y,WT,MODEL,LOSSVEC(I), &
                  LOSSD1VEC(I))
            ELSE
               CALL SIMPLE_GLM_LOSS(BETAVEC(I),X,C,Y,WT,MODEL,LOSSVEC(I), &
                  LOSSD1VEC(I))
            END IF
         END DO
         CALL PENALTY_FUN(BETAVEC,RHO,ETA,PENTYPE,PENVEC,PEND1VEC)
         NEGIDX = LOSSD1VEC+PEND1VEC<-EPS
         IF (ANY(NEGIDX)) THEN
            DOBISECTION = .TRUE.
            IDX = MAXLOC(BETAVEC,1,NEGIDX)
            XBRACKET(1) = BETAVEC(IDX)
            D1BRACKET(1) = LOSSD1VEC(IDX) + PEND1VEC(IDX)
            IDX = MINLOC(BETAVEC,1,(BETAVEC>XBRACKET(1)).AND.(LOSSD1VEC+PEND1VEC>=ZERO))
            XBRACKET(3) = BETAVEC(IDX)
            D1BRACKET(3) = LOSSD1VEC(IDX) + PEND1VEC(IDX)
         ELSE
            DOBISECTION = .FALSE.
            XMIN = ZERO
            RETURN
         END IF
      END IF
!
!     Bisection
!
      IF (DOBISECTION) THEN
         DO
            XBRACKET(2) = HALF*(XBRACKET(1)+XBRACKET(3))
            IF (ISNEGROOT) THEN
               CALL SIMPLE_GLM_LOSS(XBRACKET(2),-X,C,Y,WT,MODEL,LOSSVEC(2),LOSSD1VEC(2))
            ELSE
               CALL SIMPLE_GLM_LOSS(XBRACKET(2),X,C,Y,WT,MODEL,LOSSVEC(2),LOSSD1VEC(2))
            END IF
            CALL PENALTY_FUN(XBRACKET(2:2),RHO,ETA,PENTYPE,PENVEC(2:2),PEND1VEC(2:2))
            D1BRACKET(2) = LOSSD1VEC(2) + PEND1VEC(2)
            IF (ABS(D1BRACKET(2))<=EPS) THEN
               XMIN = XBRACKET(2)
               EXIT
            ELSEIF (D1BRACKET(2)<-EPS) THEN
               XBRACKET(1) = XBRACKET(2)
               D1BRACKET(1) = D1BRACKET(2)
            ELSE
               XBRACKET(3) = XBRACKET(2)
               D1BRACKET(3) = D1BRACKET(2)
            END IF
            IF (ABS(XBRACKET(3)-XBRACKET(1))<EPS) THEN
               XMIN = HALF*(XBRACKET(1)+XBRACKET(3))
               EXIT
            END IF
         END DO
!
!     Compare the bracket root to zero
!
         IF (LOSSVEC(1)+PENVEC(1)<LOSSVEC(2)+PENVEC(2)) THEN
            XMIN = ZERO
            RETURN
         ELSEIF (ISNEGROOT) THEN
            XMIN = -XMIN
         END IF
      END IF
      RETURN
      END FUNCTION GLM_THRESHOLDING
!
      SUBROUTINE SIMPLE_GLM_LOSS(BETA,X,C,Y,WT,MODEL,LOSS,D1,D2)
!
!     This subroutine computes the loss, derivative, and second derivative
!     of a GLM model with linear part: X*BETA+C
!
      IMPLICIT NONE
      CHARACTER(LEN=*), INTENT(IN) :: MODEL
      INTEGER :: I,N
      REAL(KIND=DBLE_PREC) :: BETA,LOSS
      REAL(KIND=DBLE_PREC), OPTIONAL :: D1,D2
      REAL(KIND=DBLE_PREC), PARAMETER :: BIG=FIVE*TEN
      REAL(KIND=DBLE_PREC), DIMENSION(:), INTENT(IN) :: C,WT,X,Y
      REAL(KIND=DBLE_PREC), DIMENSION(SIZE(X)) :: INNER,PROB
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
!     Compute loss function
!
      INNER = BETA*X+C
      SELECT CASE(MODEL)
      CASE("LOGISTIC")
         WHERE (INNER>=BIG)
            PROB = INNER
         ELSEWHERE (INNER<=-BIG)
            PROB = ZERO
         ELSEWHERE
            PROB = LOG(ONE+EXP(INNER))
         END WHERE
         LOSS = - SUM(WT*(Y*INNER-PROB))
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
               PROB = EXP(INNER)/(ONE+EXP(INNER))
            END WHERE
            D1 = - SUM(WT*(Y-PROB)*X)
         END SELECT
      END IF
!
!     Compute second derivative
!
      IF (PRESENT(D2)) THEN
         SELECT CASE(MODEL)
         CASE("LOGISTIC")
            D2 = SUM(WT*PROB*(ONE-PROB)*X*X)
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
      REAL(KIND=DBLE_PREC) :: A,B,ETA,L,M,MAXRHO,R,ROOTL,ROOTM,ROOTR
      REAL(KIND=DBLE_PREC), PARAMETER :: EPS=1E-8
      REAL(KIND=DBLE_PREC), DIMENSION(:) :: PENPARAM
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
         B = -B/A
         L = A/(A+ONE)*ABS(B)
         R = A*ABS(B)
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
      INTEGER, PARAMETER :: GRIDPTS=50
      INTEGER :: I
      LOGICAL :: ISINFRHO,ISNEGROOT
      REAL(KIND=DBLE_PREC), PARAMETER :: EPS=1E-8,MULTIPLIER=TWO
      REAL(KIND=DBLE_PREC) :: DELTABETA,ETA,L,LOSS,M,MAXRHO,R,ROOTL,ROOTM,ROOTR
      REAL(KIND=DBLE_PREC), DIMENSION(:) :: C,PENPARAM,WT,X,Y
      REAL(KIND=DBLE_PREC), DIMENSION(GRIDPTS) :: BETAGRID,LOSSGRID,PENGRID
!
!     Figure out whether max rho is infinity
!
      ETA = PENPARAM(1)
      SELECT CASE(PENTYPE)
      CASE("ENET")
         IF (ABS(ETA-TWO)<EPS) THEN
            ISINFRHO = .TRUE.
         ELSE
            ISINFRHO = .FALSE.
         END IF
      CASE("LOG")
         ISINFRHO = .FALSE.
      CASE("MCP")
         ISINFRHO = .FALSE.
      CASE("POWER")
         IF (ETA>ONE) THEN
            ISINFRHO = .TRUE.
         ELSE
            ISINFRHO = .FALSE.
         END IF
      CASE("SCAD")
         ISINFRHO = .FALSE.
      END SELECT
!
!     Compute unpenalized root
!
      L = ZERO
      ROOTL = GLM_THRESHOLDING(X,C,Y,WT,L,ETA,PENTYPE,MODEL)
      IF (ROOTL<ZERO) THEN
         ISNEGROOT = .TRUE.
      END IF
!
!     If max rho is infinity, output rho such that estimate<.1*unbiased estimate
!
      IF (ISINFRHO) THEN
         R = ONE
         DO 
            ROOTR = GLM_THRESHOLDING(X,C,Y,WT,R,ETA,PENTYPE,MODEL)
            IF (ABS(ROOTR)<=ABS(ROOTL)/TEN) THEN
               EXIT
            END IF
            R = R*MULTIPLIER
         END DO
         MAXRHO = R
         RETURN
      END IF
!
!     Crude search for right bracket
!
      DELTABETA = ROOTL/(GRIDPTS-1)
      DO I=1,GRIDPTS
         BETAGRID(I) = DELTABETA*(I-1)
         CALL SIMPLE_GLM_LOSS(BETAGRID(I),X,C,Y,WT,MODEL,LOSSGRID(I))
      END DO
      R = ONE
      DO
         CALL PENALTY_FUN(BETAGRID,R,ETA,PENTYPE,PENGRID)
         IF (ALL(LOSSGRID+PENGRID>=LOSSGRID(1)+PENGRID(1)-EPS)) THEN
            EXIT
         ELSE
            R = R*MULTIPLIER
         END IF
      END DO
      ROOTR = ZERO
!
!     Bisection to locate the max rho
!
      DO
         M = HALF*(L+R)
         ROOTM = GLM_THRESHOLDING(X,C,Y,WT,M,ETA,PENTYPE,MODEL)
         IF (ABS(ROOTM)>EPS) THEN
            L = M
            ROOTL = ROOTM
         ELSE
            R = M
            ROOTR = ROOTM
         END IF
         IF (ABS(L-R)<EPS) THEN
            MAXRHO = HALF*(L+R)
            EXIT
         END IF
      END DO
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
      INTEGER :: I,ITERATION,M,MAXITERS,N
      REAL(KIND=DBLE_PREC) :: CRITERION=1E-4,EPS=1E-8
      REAL(KIND=DBLE_PREC) :: A,B,LAMBDA,NEW_OBJECTIVE,OLDROOT
      REAL(KIND=DBLE_PREC) :: OBJECTIVE,ROOTDIFF
      LOGICAL, DIMENSION(:) :: PENIDX
      REAL(KIND=DBLE_PREC), DIMENSION(:) :: ESTIMATE,PENPARAM,SUM_X_SQUARES,WT,Y
      REAL(KIND=DBLE_PREC), DIMENSION(:,:) :: X
      LOGICAL, DIMENSION(SIZE(ESTIMATE)) :: NZIDX
      REAL(KIND=DBLE_PREC), DIMENSION(SIZE(Y)) :: R
      REAL(KIND=DBLE_PREC), DIMENSION(SIZE(ESTIMATE)) :: PENALTY
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
!     LAMBDA. Note that the rows of X correspond to cases and the columns
!     to predictors.
!
      IMPLICIT NONE
      CHARACTER(LEN=*), INTENT(IN) :: PENTYPE,MODEL
      INTEGER :: ITERATION,J,MAXITERS,N,P
      REAL(KIND=DBLE_PREC) :: CRITERION=1E-4,EPS=1E-8
      REAL(KIND=DBLE_PREC) :: LAMBDA,LOSS,NEW_OBJECTIVE,OLDROOT
      REAL(KIND=DBLE_PREC) :: OBJECTIVE,ROOTDIFF
      LOGICAL, DIMENSION(:) :: PENIDX
      REAL(KIND=DBLE_PREC), DIMENSION(:) :: ESTIMATE,PENPARAM,WT,Y
      REAL(KIND=DBLE_PREC), DIMENSION(:,:) :: X
      REAL(KIND=DBLE_PREC), DIMENSION(SIZE(Y)) :: C,INNER
      REAL(KIND=DBLE_PREC), DIMENSION(SIZE(ESTIMATE)) :: PENALTY
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
      N = SIZE(Y)
      P = SIZE(ESTIMATE)
!
!     Initialize the residual vector R, the PENALTY, the loss L2, and the
!     objective function.
!
      IF (ANY(ABS(ESTIMATE)>EPS)) THEN
         INNER = MATMUL(X,ESTIMATE)
      ELSE
         INNER = ZERO
      END IF
      C = ZERO
      CALL SIMPLE_GLM_LOSS(ONE,INNER,C,Y,WT,MODEL,LOSS)
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
         DO J = 1,P
            OLDROOT = ESTIMATE(J)
            C = INNER - X(:,J)*OLDROOT
            IF (PENIDX(J)) THEN
               ESTIMATE(J) = GLM_THRESHOLDING(X(:,J),C,Y,WT,LAMBDA,PENPARAM(1),PENTYPE,MODEL)
            ELSE
               ESTIMATE(J) = GLM_THRESHOLDING(X(:,J),C,Y,WT,ZERO,PENPARAM(1),PENTYPE,MODEL)
            END IF
            ROOTDIFF = ESTIMATE(J)-OLDROOT
            IF (ABS(ROOTDIFF)>EPS) THEN
               INNER = INNER + ROOTDIFF*X(:,J)
            END IF
         END DO
!
!     Output the iteration number and value of the objective function.
!
         C = ZERO
         CALL SIMPLE_GLM_LOSS(ONE,INNER,C,Y,WT,MODEL,LOSS)
         CALL PENALTY_FUN(ESTIMATE,LAMBDA,PENPARAM(1),PENTYPE,PENALTY)
         NEW_OBJECTIVE = LOSS+SUM(PENALTY,PENIDX)
         IF (ITERATION==1.OR.MOD(ITERATION,1)==0) THEN
            !PRINT*," ITERATION = ",ITERATION," FUN = ",NEW_OBJECTIVE
         END IF
         !PRINT*, "ESTIMATE = ", ESTIMATE         
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
      END SUBROUTINE PENALIZED_GLM_REGRESSION
!
      END MODULE SPARSEREG

      