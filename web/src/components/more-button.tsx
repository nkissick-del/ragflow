import { cn } from '@/lib/utils';
import { Ellipsis } from 'lucide-react';
import React from 'react';
import { useTranslation } from 'react-i18next';
import { Button, ButtonProps } from './ui/button';

export const MoreButton = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, size, ...props }, ref) => {
    const { t } = useTranslation();
    return (
      <Button
        aria-label={t('common.moreOptions')}
        ref={ref}
        variant="ghost"
        size={size || 'icon'}
        className={cn(
          'opacity-0 group-hover:opacity-100 focus-visible:opacity-100 size-3.5 bg-transparent group-hover:bg-transparent',
          className,
        )}
        {...props}
      >
        <Ellipsis />
      </Button>
    );
  },
);

MoreButton.displayName = 'MoreButton';
